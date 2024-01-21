// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"math"
	"math/rand"
	"sort"
)

// Optimizer is an optimizer
type Optimizer struct {
	N      int
	Length int
	Scale  float64
	Rng    *rand.Rand
	Vars   [][3]RandomMatrix
	Cost   func(samples []OptimizerSample)
}

// OptimizerSample is a sample of the optimizer
type OptimizerSample struct {
	Cost float64
	Vars [][3]Matrix
}

// NewOptimizer creates a new optimizer
func NewOptimizer(rng *rand.Rand, n int, scale float64, vars int, a Matrix,
	cost func(samples []OptimizerSample)) Optimizer {
	o := Optimizer{
		N:      n,
		Length: n * n * n,
		Scale:  scale,
		Rng:    rng,
		Cost:   cost,
	}
	mean, stddev := 0.0, 0.0
	for _, value := range a.Data {
		mean += float64(value)
	}
	mean /= float64(len(a.Data))
	for _, value := range a.Data {
		diff := mean - float64(value)
		stddev += diff * diff
	}
	stddev /= float64(len(a.Data))
	stddev = math.Sqrt(stddev)
	o.Vars = make([][3]RandomMatrix, vars)
	for v := range o.Vars {
		o.Vars[v][0] = NewRandomMatrix(a.Cols, a.Rows)
		for j := range o.Vars[v][0].Data {
			o.Vars[v][0].Data[j].Mean = 0
			o.Vars[v][0].Data[j].StdDev = mean
		}
		o.Vars[v][1] = NewRandomMatrix(a.Cols, a.Rows)
		for j := range o.Vars[v][1].Data {
			o.Vars[v][1].Data[j].Mean = 0
			o.Vars[v][1].Data[j].StdDev = math.Sqrt(stddev)
		}
		o.Vars[v][2] = NewRandomMatrix(a.Cols, a.Rows)
		for j := range o.Vars[v][2].Data {
			o.Vars[v][2].Data[j].Mean = 0
			o.Vars[v][2].Data[j].StdDev = math.Sqrt(stddev)
		}
	}
	return o
}

func (o *Optimizer) Iterate() OptimizerSample {
	samples := make([]OptimizerSample, o.Length, o.Length)
	s := make([][][]Matrix, len(o.Vars))
	for v := range s {
		s[v] = make([][]Matrix, 3)
		for j := range s[v] {
			s[v][j] = make([]Matrix, o.N)
			for k := range s[v][j] {
				s[v][j][k] = o.Vars[0][j].Sample(o.Rng)
			}
		}
	}
	for v := range s {
		index := 0
		for _, x := range s[v][0] {
			for _, y := range s[v][1] {
				for _, z := range s[v][2] {
					if samples[index].Vars == nil {
						samples[index].Vars = make([][3]Matrix, len(s), len(s))
					}
					samples[index].Vars[v][0] = x
					samples[index].Vars[v][1] = y
					samples[index].Vars[v][2] = z
					index++
				}
			}
		}
	}

	o.Cost(samples)
	sort.Slice(samples, func(i, j int) bool {
		return samples[i].Cost < samples[j].Cost
	})

	mean, stddev := 0.0, 0.0
	for i := range samples {
		mean += samples[i].Cost
	}
	mean /= float64(len(samples))
	for i := range samples {
		diff := mean - samples[i].Cost
		stddev += diff * diff
	}
	stddev /= float64(len(samples))
	stddev = math.Sqrt(stddev)

	weights, sum := make([]float64, o.Length, o.Length), 0.0
	for i := range weights {
		diff := (samples[i].Cost - mean) / stddev
		weight := math.Exp(-(diff*diff/2 + o.Scale*float64(i))) / (stddev * math.Sqrt(2*math.Pi))
		sum += weight
		weights[i] = weight
	}
	for i := range weights {
		weights[i] /= sum
	}

	for j := range o.Vars {
		for v := range o.Vars[j] {
			vv := NewRandomMatrix(o.Vars[j][v].Cols, o.Vars[j][v].Rows)
			for k := range vv.Data {
				vv.Data[k].StdDev = 0
			}
			for k := range samples {
				for l, value := range samples[k].Vars[j][v].Data {
					vv.Data[l].Mean += weights[k] * float64(value)
				}
			}
			for k := range samples {
				for l, value := range samples[k].Vars[j][v].Data {
					diff := vv.Data[l].Mean - float64(value)
					vv.Data[l].StdDev += weights[k] * diff * diff
				}
			}
			for k := range vv.Data {
				vv.Data[k].StdDev /= (float64(o.Length) - 1.0) / float64(o.Length)
				vv.Data[k].StdDev = math.Sqrt(vv.Data[k].StdDev)
			}
			o.Vars[j][v] = vv
		}
	}

	return samples[0]
}

// Optimize optimizes a cost function
func (o *Optimizer) Optimize(dx float64) OptimizerSample {
	last := -1.0
	for {
		s := o.Iterate()
		if last > 0 && math.Abs(last-s.Cost) < dx {
			return s
		}
		last = s.Cost
	}
}
