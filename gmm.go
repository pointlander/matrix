// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
)

// GMM is a gaussian mixture model
type GMM struct {
	Clusters int
	Samples  int
	Rng      *rand.Rand
}

// NewGMM return a new gmm model
func NewGMM() GMM {
	return GMM{
		Clusters: 20,
		Samples:  1024,
		Rng:      rand.New(rand.NewSource(3)),
	}
}

// GMM is a gaussian mixture model clustering algorithm
// https://github.com/Ransaka/GMM-from-scratch
// https://en.wikipedia.org/wiki/Multivariate_normal_distribution
func (g *GMM) GMM(input Matrix) []int {
	rng := g.Rng

	type Cluster struct {
		E RandomMatrix
		U RandomMatrix
	}
	var Pi RandomMatrix
	mean, stddev, count := float32(0.0), float32(0.0), float32(0.0)
	for _, value := range input.Data {
		mean += value
		count++
	}
	mean /= count
	for _, value := range input.Data {
		diff := value - mean
		stddev += diff * diff
	}
	stddev /= count
	stddev = float32(math.Sqrt(float64(stddev)))
	factor := float32(math.Sqrt(2.0 / float64(input.Cols)))
	clusters := make([]Cluster, g.Clusters, g.Clusters)
	for i := range clusters {
		clusters[i].E = NewRandomMatrix(input.Cols, input.Cols)
		for j := range clusters[i].E.Data {
			clusters[i].E.Data[j].Mean = factor * float32(rng.NormFloat64())
			clusters[i].E.Data[j].StdDev = factor * float32(rng.NormFloat64())
		}
		clusters[i].U = NewRandomMatrix(input.Cols, 1)
		for j := range clusters[i].U.Data {
			clusters[i].U.Data[j].Mean = factor * float32(rng.NormFloat64())
			clusters[i].U.Data[j].StdDev = factor * float32(rng.NormFloat64())
		}
	}
	Pi = NewRandomMatrix(g.Clusters, input.Rows)
	for j := range Pi.Data {
		Pi.Data[j].StdDev = 1
	}

	type Sample struct {
		E  []Matrix
		U  []Matrix
		C  float64
		Pi []float64
	}
	done, samples, cpus := make(chan bool, 8), make([]Sample, g.Samples, g.Samples), runtime.NumCPU()
	process := func(j int, seed int64) {
		rng := rand.New(rand.NewSource(seed))
		samples[j].Pi = make([]float64, g.Clusters*input.Rows, g.Clusters*input.Rows)
		for l := 0; l < len(samples[j].Pi); l += g.Clusters {
			sum := 0.0
			for k := range clusters {
				r := Pi.Data[l+k]
				samples[j].Pi[l+k] = float64(r.StdDev)*rng.NormFloat64() + float64(r.Mean)
				if samples[j].Pi[l+k] < 0 {
					samples[j].Pi[l+k] = -samples[j].Pi[l+k]
				}
				sum += samples[j].Pi[l+k]
			}
			for k := 0; k < g.Clusters; k++ {
				samples[j].Pi[l+k] /= sum
			}
		}
		cs := make([][]float64, g.Clusters)
		for k := range cs {
			cs[k] = make([]float64, input.Rows, input.Rows)
		}
		samples[j].E = make([]Matrix, g.Clusters)
		samples[j].U = make([]Matrix, g.Clusters)
		samples[j].C = 0
		for k := range clusters {
			samples[j].E[k] = clusters[k].E.Sample(rng)
			samples[j].U[k] = clusters[k].U.Sample(rng)

			d, _ := Determinant(samples[j].E[k])
			det := float32(d)
			for f := 0; f < input.Rows; f++ {
				row := input.Data[f*input.Cols : (f+1)*input.Cols]
				x := NewMatrix(input.Cols, 1, row...)
				y := MulT(T(MulT(Sub(x, samples[j].U[k]), samples[j].E[k])), Sub(x, samples[j].U[k]))
				pdf := math.Pow(2*math.Pi, -float64(input.Cols)/2) *
					math.Pow(float64(det), 1/2) *
					math.Exp(float64(-y.Data[0]/2))
				cs[k][f] = samples[j].Pi[f*g.Clusters+k] * pdf
			}
		}
		for f := 0; f < input.Rows; f++ {
			sum := 0.0
			for k := range clusters {
				sum += cs[k][f]
			}
			for k := range clusters {
				cs[k][f] /= sum
			}
		}
		for k := range clusters {
			mean := 0.0
			for _, value := range cs[k] {
				mean += value
			}
			mean /= float64(input.Rows)
			stddev := 0.0
			for _, value := range cs[k] {
				diff := value - mean
				stddev += diff * diff
			}
			stddev /= float64(input.Rows)
			stddev = math.Sqrt(stddev)
			samples[j].C += math.Exp(-stddev)
		}
		samples[j].C /= float64(len(clusters))
		done <- true
	}
	last := -1.0
	for {
		j, flight := 0, 0
		for j < g.Samples && flight < cpus {
			go process(j, rng.Int63())
			j++
			flight++
		}
		for j < g.Samples {
			<-done
			flight--

			go process(j, rng.Int63())
			j++
			flight++
		}
		for f := 0; f < flight; f++ {
			<-done
		}

		aa := make([]Cluster, g.Clusters, g.Clusters)
		for i := range aa {
			aa[i].E = NewRandomMatrix(input.Cols, input.Cols)
			for j := range aa[i].E.Data {
				aa[i].E.Data[j].StdDev = 0
			}
			aa[i].U = NewRandomMatrix(input.Cols, 1)
			for j := range aa[i].U.Data {
				aa[i].U.Data[j].StdDev = 0
			}
		}
		pi := NewRandomMatrix(g.Clusters, input.Rows)
		for j := range pi.Data {
			pi.Data[j].StdDev = 0
		}

		sort.Slice(samples, func(i, j int) bool {
			return samples[i].C < samples[j].C
		})
		fmt.Println(samples[0].C)

		mean, stddev := 0.0, 0.0
		for i := range samples {
			mean += float64(samples[i].C)
		}
		mean /= float64(len(samples))
		for i := range samples {
			diff := mean - float64(samples[i].C)
			stddev += diff * diff
		}
		stddev /= float64(len(samples))
		stddev = math.Sqrt(stddev)

		// https://stats.stackexchange.com/questions/6534/how-do-i-calculate-a-weighted-standard-deviation-in-excel
		weights, sum := make([]float32, g.Samples), float32(0)
		for i := range weights {
			diff := (float64(samples[i].C) - mean) / stddev
			w := float32(math.Exp(-(diff*diff/2 + float64(i))) / (stddev * math.Sqrt(2*math.Pi)))
			sum += w
			weights[i] = w
		}
		for i := range weights {
			weights[i] /= sum
		}

		for k := range clusters {
			for i := range samples {
				for x := range aa[k].E.Data {
					value := samples[i].E[k].Data[x]
					aa[k].E.Data[x].Mean += float32(weights[i]) * value
				}
			}
			for i := range samples {
				for x := range aa[k].E.Data {
					diff := aa[k].E.Data[x].Mean - samples[i].E[k].Data[x]
					aa[k].E.Data[x].StdDev += float32(weights[i]) * diff * diff
				}
			}
			for x := range aa[k].E.Data {
				aa[k].E.Data[x].StdDev /= (float32(g.Samples) - 1) / float32(g.Samples)
				aa[k].E.Data[x].StdDev = float32(math.Sqrt(float64(aa[k].E.Data[x].StdDev)))
			}

			for i := range samples {
				for x := range aa[k].U.Data {
					value := samples[i].U[k].Data[x]
					aa[k].U.Data[x].Mean += float32(weights[i]) * value
				}
			}
			for i := range samples {
				for x := range aa[k].U.Data {
					diff := aa[k].U.Data[x].Mean - samples[i].U[k].Data[x]
					aa[k].U.Data[x].StdDev += float32(weights[i]) * diff * diff
				}
			}
			for x := range aa[k].U.Data {
				aa[k].U.Data[x].StdDev /= (float32(g.Samples) - 1) / float32(g.Samples)
				aa[k].U.Data[x].StdDev = float32(math.Sqrt(float64(aa[k].U.Data[x].StdDev)))
			}
		}

		for i := range samples {
			for x := range pi.Data {
				value := samples[i].Pi[x]
				pi.Data[x].Mean += weights[i] * float32(value)
			}
		}
		for i := range samples {
			for x := range pi.Data {
				diff := float32(pi.Data[x].Mean) - float32(samples[i].Pi[x])
				pi.Data[x].StdDev += float32(weights[i]) * diff * diff
			}
		}
		for x := range pi.Data {
			pi.Data[x].StdDev /= (float32(g.Samples) - 1) / float32(g.Samples)
			pi.Data[x].StdDev = float32(math.Sqrt(float64(pi.Data[x].StdDev)))
		}

		clusters = aa
		Pi = pi

		if last > 0 && math.Abs(last-samples[0].C) < 1e-6 {
			break
		}
		last = samples[0].C
	}

	sort.Slice(samples, func(i, j int) bool {
		return samples[i].C < samples[j].C
	})
	sample := samples[0]

	output := make([]int, input.Rows)
	for i := 0; i < input.Rows; i++ {
		row := input.Data[i*input.Cols : (i+1)*input.Cols]
		x := NewMatrix(input.Cols, 1, row...)

		index, max := 0, 0.0
		for j := 0; j < g.Clusters; j++ {
			d, _ := Determinant(sample.E[j])
			det := float32(d)
			y := MulT(T(MulT(Sub(x, sample.U[j]), sample.E[j])), Sub(x, sample.U[j]))
			pdf := math.Pow(2*math.Pi, -float64(input.Cols)/2) *
				math.Pow(float64(det), 1/2) *
				math.Exp(float64(-y.Data[0]/2))
			pdf *= float64(samples[j].Pi[i*g.Clusters+j])
			if pdf > max {
				index, max = j, pdf
			}
		}
		output[i] = index
	}
	return output
}
