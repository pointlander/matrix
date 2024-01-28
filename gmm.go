// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"math"
	"math/rand"
	"runtime"
)

// GMMO is an optimizer based gmm
type GMM struct {
	Optimizer
	Clusters int
}

// NewGMM creates a new optimizer based gmm
// https://github.com/Ransaka/GMM-from-scratch
// https://en.wikipedia.org/wiki/Multivariate_normal_distribution
func NewGMM(input Matrix, clusters int) GMM {
	const n = 16
	o := Optimizer{
		N:      n,
		Length: n * n * n,
		Scale:  .01,
		Rng:    rand.New(rand.NewSource(3)),
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
					det, _ := Determinant(E)
					for f := 0; f < input.Rows; f++ {
						row := input.Data[f*input.Cols : (f+1)*input.Cols]
						x := NewMatrix(input.Cols, 1, row...)
						y := MulT(T(MulT(Sub(x, U), E)), Sub(x, U))
						pdf := math.Pow(2*math.Pi, -float64(input.Cols)/2) *
							math.Sqrt(math.Abs(det)) *
							math.Exp(float64(-y.Data[0])/2)
						cs[k][f] = float64(samples[j].Vars[2*clusters][0].Data[f*clusters+k]) * pdf
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
	o.Vars = make([][3]RandomMatrix, 2*clusters+1)
	for v := range o.Vars[:clusters] {
		o.Vars[v][0] = NewRandomMatrix(input.Cols, input.Cols)
		for j := range o.Vars[v][0].Data {
			o.Vars[v][0].Data[j].Mean = 0
			o.Vars[v][0].Data[j].StdDev = mean
		}
		o.Vars[v][1] = NewRandomMatrix(input.Cols, input.Cols)
		for j := range o.Vars[v][1].Data {
			o.Vars[v][1].Data[j].Mean = 0
			o.Vars[v][1].Data[j].StdDev = mean
		}
		o.Vars[v][2] = NewRandomMatrix(input.Cols, input.Cols)
		for j := range o.Vars[v][2].Data {
			o.Vars[v][2].Data[j].Mean = 0
			o.Vars[v][2].Data[j].StdDev = mean
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
			u[v][1].Data[j].StdDev = mean
		}
		u[v][2] = NewRandomMatrix(input.Cols, 1)
		for j := range u[v][2].Data {
			u[v][2].Data[j].Mean = 0
			u[v][2].Data[j].StdDev = mean
		}
	}
	for v := range o.Vars[2*clusters] {
		o.Vars[2*clusters][v] = NewRandomMatrix(clusters, input.Rows)
		for j := range o.Vars[2*clusters][v].Data {
			o.Vars[2*clusters][v].Data[j].StdDev = 1
		}
	}

	return GMM{
		Optimizer: o,
		Clusters:  clusters,
	}
}

func (g *GMM) Optimize(input Matrix) []int {
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
				math.Sqrt(math.Abs(det)) *
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

func MetaGMM(input Matrix, clusters int) []int {
	rng := rand.New(rand.NewSource(3))
	a := make([]Matrix, 2*clusters+1, 2*clusters+1)
	cluster := 0
	for cluster < clusters {
		a[cluster] = NewCoord(input.Cols, input.Cols)
		cluster++
	}
	for cluster < 2*clusters {
		a[cluster] = NewCoord(input.Cols, 1)
		cluster++
	}
	a[cluster] = NewCoord(clusters, input.Rows)
	sample := Meta(256, .082, .1, rng, 4, .1, 2*clusters+1, func(samples []Sample, a ...Matrix) {
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
				det, _ := Determinant(E)
				for f := 0; f < input.Rows; f++ {
					row := input.Data[f*input.Cols : (f+1)*input.Cols]
					x := NewMatrix(input.Cols, 1, row...)
					y := MulT(T(MulT(Sub(x, U), E)), Sub(x, U))
					pdf := math.Pow(2*math.Pi, -float64(input.Cols)/2) *
						math.Sqrt(math.Abs(det)) *
						math.Exp(float64(-y.Data[0])/2)
					cs[k][f] = float64(samples[j].Vars[2*clusters][0].Data[f*clusters+k]) * pdf
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
	}, a...)

	output := make([]int, input.Rows)
	for i := 0; i < input.Rows; i++ {
		row := input.Data[i*input.Cols : (i+1)*input.Cols]
		x := NewMatrix(input.Cols, 1, row...)

		index, max := 0, 0.0
		for j := 0; j < clusters; j++ {
			E := Add(sample.Vars[j][0], H(sample.Vars[j][1], sample.Vars[j][2]))
			U := Add(sample.Vars[j+clusters][0], H(sample.Vars[j+clusters][1], sample.Vars[j+clusters][2]))
			det, _ := Determinant(E)
			y := MulT(T(MulT(Sub(x, U), E)), Sub(x, U))
			pdf := math.Pow(2*math.Pi, -float64(input.Cols)/2) *
				math.Sqrt(math.Abs(det)) *
				math.Exp(float64(-y.Data[0])/2)
			pdf *= float64(sample.Vars[2*clusters][0].Data[i*clusters+j])
			if pdf > max {
				index, max = j, pdf
			}
		}
		output[i] = index
	}
	return output
}
