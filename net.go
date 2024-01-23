// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"math/rand"
	"runtime"
)

// Net is a net
type Net struct {
	Optimizer
}

// NewNet makes a new network
func NewNet(seed int64, inputs, outputs int) Net {
	rng := rand.New(rand.NewSource(seed))
	return Net{
		Optimizer: NewOptimizer(rng, 8, .01, 3, func(samples []Sample, a ...Matrix) {
			q := NewZeroMatrix(outputs, len(samples))
			k := NewZeroMatrix(outputs, len(samples))
			v := NewZeroMatrix(outputs, len(samples))
			done, cpus := make(chan bool, 8), runtime.NumCPU()
			process := func(i int) {
				{
					x := samples[i].Vars[0][0]
					y := samples[i].Vars[0][1]
					z := samples[i].Vars[0][2]
					neurons := Add(x, H(y, z))
					query := MulT(neurons, a[0])
					copy(q.Data[i*outputs:], query.Data)
				}
				{
					x := samples[i].Vars[1][0]
					y := samples[i].Vars[1][1]
					z := samples[i].Vars[1][2]
					key := MulT(Add(x, H(y, z)), a[1])
					copy(k.Data[i*outputs:], key.Data)
				}
				{
					x := samples[i].Vars[1][0]
					y := samples[i].Vars[1][1]
					z := samples[i].Vars[1][2]
					value := MulT(Add(x, H(y, z)), a[2])
					copy(v.Data[i*outputs:], value.Data)
				}
				done <- true
			}
			j, flight := 0, 0
			for j < len(samples) && flight < cpus {
				go process(j)
				j++
				flight++
			}
			for j < len(samples) {
				<-done
				flight--

				go process(j)
				j++
				flight++
			}
			for f := 0; f < flight; f++ {
				<-done
			}

			entropies := SelfEntropy64(q, k, v)
			for i, entropy := range entropies {
				samples[i].Cost = entropy
			}
		}, NewCoord(inputs, outputs)),
	}
}

// Fire runs the network
func (n *Net) Fire(query, key, value Matrix) (float64, Matrix, Matrix, Matrix) {
	s := n.Iterate(query, key, value)
	q := MulT(Add(s.Vars[0][0], H(s.Vars[0][1], s.Vars[0][2])), query)
	k := MulT(Add(s.Vars[1][0], H(s.Vars[1][1], s.Vars[1][2])), key)
	v := MulT(Add(s.Vars[2][0], H(s.Vars[2][1], s.Vars[2][2])), value)
	return s.Cost, q, k, v
}
