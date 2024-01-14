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
	Out     Matrix
}

// CalculateStatistics calculates the statistics of systems
func (n Net) CalculateStatistics(systems []Sample) RandomMatrix {
	statistics := NewRandomMatrix(n.Inputs, n.Outputs)
	for i := range statistics.Data {
		statistics.Data[i].StdDev = 0
	}
	weights, sum := make([]float64, n.Window), 0.0
	for i := range weights {
		weight := math.Exp(-float64(systems[i].Entropy))
		sum += weight
		weights[i] = weight
	}
	for i := range weights {
		weights[i] /= sum
	}

	for i := range systems[:n.Window] {
		for j, value := range systems[i].Neurons.Data {
			statistics.Data[j].Mean += float32(weights[i]) * value
		}
	}
	for i := range systems[:n.Window] {
		for j, value := range systems[i].Neurons.Data {
			diff := statistics.Data[j].Mean - value
			statistics.Data[j].StdDev += float32(weights[i]) * diff * diff
		}
	}
	for i := range statistics.Data {
		statistics.Data[i].StdDev /= (float32(n.Window) - 1.0) / float32(n.Window)
		statistics.Data[i].StdDev = float32(math.Sqrt(float64(statistics.Data[i].StdDev)))
	}
	return statistics
}

// Fire runs the network
func (n *Net) Fire(query, key, value Matrix) (float32, Matrix, Matrix, Matrix) {
	q := NewMatrix(n.Outputs, n.Samples)
	q.Data = q.Data[:n.Outputs*n.Samples]
	k := NewMatrix(n.Outputs, n.Samples)
	k.Data = k.Data[:n.Outputs*n.Samples]
	v := NewMatrix(n.Outputs, n.Samples)
	v.Data = v.Data[:n.Outputs*n.Samples]
	systemsQ := make([]Sample, n.Samples)
	systemsK := make([]Sample, n.Samples)
	systemsV := make([]Sample, n.Samples)
	done, cpus := make(chan bool, 8), runtime.NumCPU()
	process := func(i int, seed int64) {
		rng := rand.New(rand.NewSource(seed))
		{
			neurons := n.Q.SampleDiscrete(rng)
			outputs := NewMatrix(n.Outputs, 1)
			out := MulT(neurons, query)
			copy(q.Data[i*n.Outputs:], out.Data)
			outputs.Data = append(outputs.Data, out.Data...)
			systemsQ[i] = Sample{
				Neurons: neurons,
				Outputs: outputs,
			}
		}
		{
			neurons := n.K.SampleDiscrete(rng)
			outputs := NewMatrix(n.Outputs, 1)
			out := MulT(neurons, key)
			copy(k.Data[i*n.Outputs:], out.Data)
			outputs.Data = append(outputs.Data, out.Data...)
			systemsK[i] = Sample{
				Neurons: neurons,
				Outputs: outputs,
			}
		}
		{
			neurons := n.V.SampleDiscrete(rng)
			outputs := NewMatrix(n.Outputs, 1)
			out := MulT(neurons, value)
			copy(v.Data[i*n.Outputs:], out.Data)
			outputs.Data = append(outputs.Data, out.Data...)
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

	outputs, entropies := SelfEntropy(q, k, v)
	for i, entropy := range entropies {
		systemsQ[i].Entropy = entropy
		systemsQ[i].Out = outputs[i]
		systemsK[i].Entropy = entropy
		systemsK[i].Out = outputs[i]
		systemsV[i].Entropy = entropy
		systemsV[i].Out = outputs[i]
	}
	sort.Slice(systemsQ, func(i, j int) bool {
		return systemsQ[i].Entropy < systemsQ[j].Entropy
	})
	sort.Slice(systemsK, func(i, j int) bool {
		return systemsK[i].Entropy < systemsK[j].Entropy
	})
	sort.Slice(systemsV, func(i, j int) bool {
		return systemsV[i].Entropy < systemsV[j].Entropy
	})

	n.Q = n.CalculateStatistics(systemsQ)
	n.K = n.CalculateStatistics(systemsK)
	n.V = n.CalculateStatistics(systemsV)

	return systemsV[0].Entropy, systemsQ[0].Outputs, systemsK[0].Outputs, systemsV[0].Outputs
}
