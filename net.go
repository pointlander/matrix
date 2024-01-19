// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"math"
	"math/rand"
	"runtime"
	"sort"
)

// Net is a net
type Net struct {
	Window  int
	Samples int
	Inputs  int
	Outputs int
	Rng     *rand.Rand
	Q       RandomMatrix
	K       RandomMatrix
	V       RandomMatrix
}

// NewNet makes a new network
func NewNet(seed int64, inputs, outputs int) Net {
	rng := rand.New(rand.NewSource(seed))
	return Net{
		Window:  32,
		Samples: 256,
		Inputs:  inputs,
		Outputs: outputs,
		Rng:     rng,
		Q:       NewRandomMatrix(inputs, outputs),
		K:       NewRandomMatrix(inputs, outputs),
		V:       NewRandomMatrix(inputs, outputs),
	}
}

// Sample is a sample of a random neural network
type Sample struct {
	Entropy float32
	Neurons Matrix
	Outputs Matrix
}

// CalculateStatistics calculates the statistics of systems
func (n Net) CalculateStatistics(mean, stddev float64, systems []Sample) RandomMatrix {
	statistics := NewRandomMatrix(n.Inputs, n.Outputs)
	for i := range statistics.Data {
		statistics.Data[i].StdDev = 0
	}

	if stddev > 0 {
		weights, sum := make([]float32, n.Samples), float32(0)
		for i := range weights {
			diff := (float64(systems[i].Entropy) - mean) / stddev
			w := float32(math.Exp(-(diff*diff/2 + .1*float64(i))) / (stddev * math.Sqrt(2*math.Pi)))
			sum += w
			weights[i] = w
		}
		for i := range weights {
			weights[i] /= sum
		}

		for i := range systems {
			for j, value := range systems[i].Neurons.Data {
				statistics.Data[j].Mean += float32(weights[i]) * value
			}
		}
		for i := range systems {
			for j, value := range systems[i].Neurons.Data {
				diff := statistics.Data[j].Mean - value
				statistics.Data[j].StdDev += float32(weights[i]) * diff * diff
			}
		}
		for i := range statistics.Data {
			statistics.Data[i].StdDev /= (float32(n.Samples) - 1.0) / float32(n.Samples)
			statistics.Data[i].StdDev = float32(math.Sqrt(float64(statistics.Data[i].StdDev)))
		}
	} else {
		s := make([]int, n.Window)
		for i := range s {
			s[i] = n.Rng.Intn(n.Samples)
		}
		for _, i := range s {
			for j, value := range systems[i].Neurons.Data {
				statistics.Data[j].Mean += value / float32(n.Window)
			}
		}
		for _, i := range s {
			for j, value := range systems[i].Neurons.Data {
				diff := statistics.Data[j].Mean - value
				statistics.Data[j].StdDev += diff * diff / float32(n.Window)
			}
		}
		for i := range statistics.Data {
			statistics.Data[i].StdDev /= (float32(n.Window) - 1.0) / float32(n.Window)
			statistics.Data[i].StdDev = float32(math.Sqrt(float64(statistics.Data[i].StdDev)))
		}
	}
	return statistics
}

// Fire runs the network
func (n *Net) Fire(query, key, value Matrix) (float32, Matrix, Matrix, Matrix) {
	q := NewZeroMatrix(n.Outputs, n.Samples)
	k := NewZeroMatrix(n.Outputs, n.Samples)
	v := NewZeroMatrix(n.Outputs, n.Samples)
	systemsQ := make([]Sample, n.Samples)
	systemsK := make([]Sample, n.Samples)
	systemsV := make([]Sample, n.Samples)
	done, cpus := make(chan bool, 8), runtime.NumCPU()
	process := func(i int, seed int64) {
		rng := rand.New(rand.NewSource(seed))
		{
			neurons := n.Q.Sample(rng)
			outputs := MulT(neurons, query)
			copy(q.Data[i*n.Outputs:], outputs.Data)
			systemsQ[i] = Sample{
				Neurons: neurons,
				Outputs: outputs,
			}
		}
		{
			neurons := n.K.Sample(rng)
			outputs := MulT(neurons, key)
			copy(k.Data[i*n.Outputs:], outputs.Data)
			systemsK[i] = Sample{
				Neurons: neurons,
				Outputs: outputs,
			}
		}
		{
			neurons := n.V.Sample(rng)
			outputs := MulT(neurons, value)
			copy(v.Data[i*n.Outputs:], outputs.Data)
			systemsV[i] = Sample{
				Neurons: neurons,
				Outputs: outputs,
			}
		}
		done <- true
	}
	j, flight := 0, 0
	for j < n.Samples && flight < cpus {
		go process(j, n.Rng.Int63()+1)
		j++
		flight++
	}
	for j < n.Samples {
		<-done
		flight--

		go process(j, n.Rng.Int63()+1)
		j++
		flight++
	}
	for f := 0; f < flight; f++ {
		<-done
	}

	entropies := SelfEntropy(q, k, v)
	for i, entropy := range entropies {
		systemsQ[i].Entropy = entropy
		systemsK[i].Entropy = entropy
		systemsV[i].Entropy = entropy
	}

	mean, stddev := 0.0, 0.0
	for _, entropy := range entropies {
		mean += float64(entropy)
	}
	mean /= float64(len(entropies))
	for _, entropy := range entropies {
		diff := mean - float64(entropy)
		stddev += diff * diff
	}
	stddev /= float64(len(entropies))
	stddev = math.Sqrt(stddev)

	if stddev > 0 {
		sort.Slice(systemsQ, func(i, j int) bool {
			return systemsQ[i].Entropy < systemsQ[j].Entropy
		})
		sort.Slice(systemsK, func(i, j int) bool {
			return systemsK[i].Entropy < systemsK[j].Entropy
		})
		sort.Slice(systemsV, func(i, j int) bool {
			return systemsV[i].Entropy < systemsV[j].Entropy
		})
	}

	n.Q = n.CalculateStatistics(mean, stddev, systemsQ)
	n.K = n.CalculateStatistics(mean, stddev, systemsK)
	n.V = n.CalculateStatistics(mean, stddev, systemsV)

	return systemsV[0].Entropy, systemsQ[0].Outputs, systemsK[0].Outputs, systemsV[0].Outputs
}
