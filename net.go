// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"runtime"
)

// Net is a net
type Net struct {
	Optimizer
}

// NewNet makes a new network
func NewNet(seed uint32, inputs, outputs int) Net {
	rng := Rand(seed)
	optimizer := NewOptimizer(&rng, 10, .1, 3, func(samples []Sample, a ...Matrix) {
		q := NewZeroMatrix(outputs, len(samples))
		k := NewZeroMatrix(outputs, len(samples))
		v := NewZeroMatrix(outputs, len(samples))
		done, cpus := make(chan bool, 8), runtime.NumCPU()
		process := func(i int) {
			{
				x := samples[i].Vars[0][0].Sample()
				y := samples[i].Vars[0][1].Sample()
				z := samples[i].Vars[0][2].Sample()
				neurons := x.Add(y.H(z))
				query := neurons.MulT(a[0])
				copy(q.Data[i*outputs:], query.Data)
			}
			{
				x := samples[i].Vars[1][0].Sample()
				y := samples[i].Vars[1][1].Sample()
				z := samples[i].Vars[1][2].Sample()
				key := x.Add(y.H(z)).MulT(a[1])
				copy(k.Data[i*outputs:], key.Data)
			}
			{
				x := samples[i].Vars[2][0].Sample()
				y := samples[i].Vars[2][1].Sample()
				z := samples[i].Vars[2][2].Sample()
				value := x.Add(y.H(z)).MulT(a[2])
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
	}, NewCoord(inputs, outputs))
	optimizer.Norm = true

	return Net{
		Optimizer: optimizer,
	}
}

// Fire runs the network
func (n *Net) Fire(query, key, value Matrix) (float64, Matrix, Matrix, Matrix) {
	s := n.Iterate(query, key, value)
	q := s.Vars[0][0].Sample().Add(s.Vars[0][1].Sample().H(s.Vars[0][2].Sample())).MulT(query)
	k := s.Vars[1][0].Sample().Add(s.Vars[1][1].Sample().H(s.Vars[1][2].Sample())).MulT(key)
	v := s.Vars[2][0].Sample().Add(s.Vars[2][1].Sample().H(s.Vars[2][2].Sample())).MulT(value)
	return s.Cost, q, k, v
}
