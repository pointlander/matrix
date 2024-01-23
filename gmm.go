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

// GMMOptimizer is an optimizer based gmm
type GMMOptimizer struct {
	Optimizer
	Clusters int
}

// NewGMMOptimizer creates a new optimizer based gmm
func NewGMMOptimizer(input Matrix, clusters int) GMMOptimizer {
	rng := rand.New(rand.NewSource(3))
	vars := 2*clusters + 1
	const (
		n = 16
	)
	o := Optimizer{
		N:      n,
		Length: n * n * n,
		Scale:  1,
		Rng:    rng,
		Cost: func(samples []Sample, a ...Matrix) {
			done, cpus := make(chan bool, 8), runtime.NumCPU()
			process := func(j int) {
				for l := range samples[j].Vars[2*clusters] {
					sum := 0.0
					for m := range samples[j].Vars[2*clusters][l].Data {
						if samples[j].Vars[2*clusters][l].Data[m] < 0 {
							samples[j].Vars[2*clusters][l].Data[m] = -samples[j].Vars[2*clusters][l].Data[m]
						}
						sum += float64(samples[j].Vars[2*clusters][l].Data[m])
					}
					for m := range samples[j].Vars[2*clusters][l].Data {
						samples[j].Vars[2*clusters][l].Data[m] /= float32(sum)
					}
				}
				cs := make([][]float64, clusters)
				for k := range cs {
					cs[k] = make([]float64, input.Rows, input.Rows)
				}
				for k := 0; k < clusters; k++ {
					E := Add(samples[j].Vars[k][0], H(samples[j].Vars[k][1], samples[j].Vars[k][2]))
					U := Add(samples[j].Vars[k+clusters][0], H(samples[j].Vars[k+clusters][1], samples[j].Vars[k+clusters][2]))
					fmt.Println(U)
					det, _ := Determinant(E)
					for f := 0; f < input.Rows; f++ {
						row := input.Data[f*input.Cols : (f+1)*input.Cols]
						x := NewMatrix(input.Cols, 1, row...)
						y := MulT(T(MulT(Sub(x, U), E)), Sub(x, U))
						pdf := math.Pow(2*math.Pi, -float64(input.Cols)/2) *
							math.Pow(det, 1/2) *
							math.Exp(float64(-y.Data[0])/2)
						cs[k][f] = float64(samples[j].Vars[clusters][0].Data[f*clusters+k]) * pdf
					}
				}
				for f := 0; f < input.Rows; f++ {
					sum := 0.0
					for k := 0; k < clusters; k++ {
						sum += cs[k][f]
					}
					for k := 0; k < clusters; k++ {
						cs[k][f] /= sum
					}
				}
				for k := 0; k < clusters; k++ {
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
					samples[j].Cost += stddev
				}
				samples[j].Cost = math.Exp(-samples[j].Cost)
				samples[j].Cost /= float64(clusters)
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
		},
	}

	mean, stddev, count := 0.0, 0.0, 0.0
	for _, value := range input.Data {
		mean += float64(value)
		count++
	}
	mean /= count
	for _, value := range input.Data {
		diff := float64(value) - mean
		stddev += diff * diff
	}
	o.Vars = make([][3]RandomMatrix, vars)
	for v := range o.Vars[:clusters] {
		o.Vars[v][0] = NewRandomMatrix(input.Cols, input.Cols)
		for j := range o.Vars[v][0].Data {
			o.Vars[v][0].Data[j].Mean = 0
			o.Vars[v][0].Data[j].StdDev = mean
		}
		o.Vars[v][1] = NewRandomMatrix(input.Cols, input.Cols)
		for j := range o.Vars[v][1].Data {
			o.Vars[v][1].Data[j].Mean = 0
			o.Vars[v][1].Data[j].StdDev = math.Sqrt(stddev)
		}
		o.Vars[v][2] = NewRandomMatrix(input.Cols, input.Cols)
		for j := range o.Vars[v][2].Data {
			o.Vars[v][2].Data[j].Mean = 0
			o.Vars[v][2].Data[j].StdDev = math.Sqrt(stddev)
		}
	}
	u := o.Vars[clusters : 2*clusters]
	for v := range u {
		u[v][0] = NewRandomMatrix(input.Cols, 1)
		for j := range u[v][0].Data {
			u[v][0].Data[j].Mean = 0
			u[v][0].Data[j].StdDev = mean
		}
		u[v][1] = NewRandomMatrix(input.Cols, 1)
		for j := range u[v][1].Data {
			u[v][1].Data[j].Mean = 0
			u[v][1].Data[j].StdDev = math.Sqrt(stddev)
		}
		u[v][2] = NewRandomMatrix(input.Cols, 1)
		for j := range u[v][2].Data {
			u[v][2].Data[j].Mean = 0
			u[v][2].Data[j].StdDev = math.Sqrt(stddev)
		}
	}
	for k := 0; k < clusters; k++ {
		for l := 0; l < 3; l++ {
			fmt.Println(o.Vars[k+clusters][l].Cols, o.Vars[k+clusters][l].Rows)
		}
	}
	for v := range o.Vars[2*clusters] {
		o.Vars[2*clusters][v] = NewRandomMatrix(clusters, input.Rows)
		for j := range o.Vars[2*clusters][v].Data {
			o.Vars[2*clusters][v].Data[j].StdDev = 1
		}
	}

	return GMMOptimizer{
		Optimizer: o,
		Clusters:  20,
	}
}

func (g *GMMOptimizer) Optimize(input Matrix) []int {
	sample := g.Optimizer.Optimize(1e-6)

	output := make([]int, input.Rows)
	for i := 0; i < input.Rows; i++ {
		row := input.Data[i*input.Cols : (i+1)*input.Cols]
		x := NewMatrix(input.Cols, 1, row...)

		index, max := 0, 0.0
		for j := 0; j < g.Clusters; j++ {
			E := Add(sample.Vars[j][0], H(sample.Vars[j][1], sample.Vars[j][2]))
			U := Add(sample.Vars[j+g.Clusters][0], H(sample.Vars[j+g.Clusters][1], sample.Vars[j+g.Clusters][2]))
			det, _ := Determinant(E)
			y := MulT(T(MulT(Sub(x, U), E)), Sub(x, U))
			pdf := math.Pow(2*math.Pi, -float64(input.Cols)/2) *
				math.Pow(det, 1/2) *
				math.Exp(float64(-y.Data[0])/2)
			pdf *= float64(sample.Vars[2*g.Clusters][0].Data[i*g.Clusters+j])
			if pdf > max {
				index, max = j, pdf
			}
		}
		output[i] = index
	}

	return output
}

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
	mean, stddev, count := 0.0, 0.0, 0.0
	for _, value := range input.Data {
		mean += float64(value)
		count++
	}
	mean /= count
	for _, value := range input.Data {
		diff := float64(value) - mean
		stddev += diff * diff
	}
	stddev /= count
	stddev = math.Sqrt(stddev)
	factor := math.Sqrt(2.0 / float64(input.Cols))
	_ = factor
	clusters := make([]Cluster, g.Clusters, g.Clusters)
	for i := range clusters {
		clusters[i].E = NewRandomMatrix(input.Cols, input.Cols)
		for j := range clusters[i].E.Data {
			clusters[i].E.Data[j].Mean = 0
			clusters[i].E.Data[j].StdDev = mean
		}
		clusters[i].U = NewRandomMatrix(input.Cols, 1)
		for j := range clusters[i].U.Data {
			clusters[i].U.Data[j].Mean = 0
			clusters[i].U.Data[j].StdDev = mean
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

			det, _ := Determinant(samples[j].E[k])
			for f := 0; f < input.Rows; f++ {
				row := input.Data[f*input.Cols : (f+1)*input.Cols]
				x := NewMatrix(input.Cols, 1, row...)
				y := MulT(T(MulT(Sub(x, samples[j].U[k]), samples[j].E[k])), Sub(x, samples[j].U[k]))
				pdf := math.Pow(2*math.Pi, -float64(input.Cols)/2) *
					math.Pow(det, 1/2) *
					math.Exp(float64(-y.Data[0])/2)
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
			samples[j].C += stddev
		}
		samples[j].C = math.Exp(-samples[j].C)
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

		mean, stddev := 0.0, 0.0
		for i := range samples {
			mean += samples[i].C
		}
		mean /= float64(len(samples))
		for i := range samples {
			diff := mean - samples[i].C
			stddev += diff * diff
		}
		stddev /= float64(len(samples))
		stddev = math.Sqrt(stddev)

		// https://stats.stackexchange.com/questions/6534/how-do-i-calculate-a-weighted-standard-deviation-in-excel
		weights, sum := make([]float64, g.Samples), 0.0
		for i := range weights {
			diff := (samples[i].C - mean) / stddev
			w := math.Exp(-(diff*diff/2 + float64(i))) / (stddev * math.Sqrt(2*math.Pi))
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
					aa[k].E.Data[x].Mean += weights[i] * float64(value)
				}
			}
			for i := range samples {
				for x := range aa[k].E.Data {
					diff := aa[k].E.Data[x].Mean - float64(samples[i].E[k].Data[x])
					aa[k].E.Data[x].StdDev += weights[i] * diff * diff
				}
			}
			for x := range aa[k].E.Data {
				aa[k].E.Data[x].StdDev /= (float64(g.Samples) - 1) / float64(g.Samples)
				aa[k].E.Data[x].StdDev = math.Sqrt(aa[k].E.Data[x].StdDev)
			}

			for i := range samples {
				for x := range aa[k].U.Data {
					value := samples[i].U[k].Data[x]
					aa[k].U.Data[x].Mean += weights[i] * float64(value)
				}
			}
			for i := range samples {
				for x := range aa[k].U.Data {
					diff := aa[k].U.Data[x].Mean - float64(samples[i].U[k].Data[x])
					aa[k].U.Data[x].StdDev += weights[i] * diff * diff
				}
			}
			for x := range aa[k].U.Data {
				aa[k].U.Data[x].StdDev /= (float64(g.Samples) - 1) / float64(g.Samples)
				aa[k].U.Data[x].StdDev = math.Sqrt(aa[k].U.Data[x].StdDev)
			}
		}

		for i := range samples {
			for x := range pi.Data {
				value := samples[i].Pi[x]
				pi.Data[x].Mean += weights[i] * float64(value)
			}
		}
		for i := range samples {
			for x := range pi.Data {
				diff := pi.Data[x].Mean - float64(samples[i].Pi[x])
				pi.Data[x].StdDev += weights[i] * diff * diff
			}
		}
		for x := range pi.Data {
			pi.Data[x].StdDev /= (float64(g.Samples) - 1) / float64(g.Samples)
			pi.Data[x].StdDev = math.Sqrt(pi.Data[x].StdDev)
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
			det, _ := Determinant(sample.E[j])
			y := MulT(T(MulT(Sub(x, sample.U[j]), sample.E[j])), Sub(x, sample.U[j]))
			pdf := math.Pow(2*math.Pi, -float64(input.Cols)/2) *
				math.Pow(det, 1/2) *
				math.Exp(float64(-y.Data[0])/2)
			pdf *= samples[j].Pi[i*g.Clusters+j]
			if pdf > max {
				index, max = j, pdf
			}
		}
		output[i] = index
	}
	return output
}
