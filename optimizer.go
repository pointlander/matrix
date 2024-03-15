// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"math"
	"runtime"
	"sort"
)

// Optimizer is an optimizer
type Optimizer struct {
	N      int
	Length int
	Scale  float64
	Rng    *Rand
	Vars   [][3]RandomMatrix
	Cost   func(samples []Sample, a ...Matrix)
	Reg    bool
	Norm   bool
}

// Sample is a sample of the optimizer
type Sample struct {
	Cost float64
	Vars [][3]Generator
}

// NewOptimizer creates a new optimizer
func NewOptimizer(rng *Rand, n int, scale float64, vars int,
	cost func(samples []Sample, a ...Matrix), a ...Matrix) Optimizer {
	o := Optimizer{
		N:      n,
		Length: n * n * n,
		Scale:  scale,
		Rng:    rng,
		Cost:   cost,
	}
	if len(a[0].Data) > 0 {
		mean, stddev := 0.0, 0.0
		for _, value := range a[0].Data {
			mean += float64(value)
		}
		mean /= float64(len(a[0].Data))
		for _, value := range a[0].Data {
			diff := mean - float64(value)
			stddev += diff * diff
		}
		stddev /= float64(len(a[0].Data))
		stddev = math.Sqrt(stddev)
		o.Vars = make([][3]RandomMatrix, vars)
		for v := range o.Vars {
			o.Vars[v][0] = NewRandomMatrix(a[0].Cols, a[0].Rows)
			for j := range o.Vars[v][0].Data {
				o.Vars[v][0].Data[j].Mean = 0
				o.Vars[v][0].Data[j].StdDev = mean
			}
			o.Vars[v][1] = NewRandomMatrix(a[0].Cols, a[0].Rows)
			for j := range o.Vars[v][1].Data {
				o.Vars[v][1].Data[j].Mean = 0
				o.Vars[v][1].Data[j].StdDev = math.Sqrt(stddev)
			}
			o.Vars[v][2] = NewRandomMatrix(a[0].Cols, a[0].Rows)
			for j := range o.Vars[v][2].Data {
				o.Vars[v][2].Data[j].Mean = 0
				o.Vars[v][2].Data[j].StdDev = math.Sqrt(stddev)
			}
		}
	} else if len(a) == 1 {
		o.Vars = make([][3]RandomMatrix, vars)
		for v := range o.Vars {
			o.Vars[v][0] = NewRandomMatrix(a[0].Cols, a[0].Rows)
			o.Vars[v][1] = NewRandomMatrix(a[0].Cols, a[0].Rows)
			o.Vars[v][2] = NewRandomMatrix(a[0].Cols, a[0].Rows)
		}
	} else {
		o.Vars = make([][3]RandomMatrix, vars)
		for v := range o.Vars {
			o.Vars[v][0] = NewRandomMatrix(a[v].Cols, a[v].Rows)
			o.Vars[v][1] = NewRandomMatrix(a[v].Cols, a[v].Rows)
			o.Vars[v][2] = NewRandomMatrix(a[v].Cols, a[v].Rows)
		}
	}
	return o
}

func (o *Optimizer) Iterate(a ...Matrix) Sample {
	samples := make([]Sample, o.Length, o.Length)
	if o.Norm {
		for i := range samples {
			samples[i].Vars = make([][3]Generator, len(o.Vars), len(o.Vars))
			for v := range samples[i].Vars {
				for j := range samples[i].Vars[v] {
					samples[i].Vars[v][j] = o.Vars[v][j].Sample(o.Rng)
				}
			}
		}
	} else {
		s := make([][][]Generator, len(o.Vars))
		for v := range s {
			s[v] = make([][]Generator, 3)
			for j := range s[v] {
				s[v][j] = make([]Generator, o.N)
				for k := range s[v][j] {
					s[v][j][k] = o.Vars[v][j].Sample(o.Rng)
				}
			}
		}
		for v := range s {
			index := 0
			for _, x := range s[v][0] {
				for _, y := range s[v][1] {
					for _, z := range s[v][2] {
						if samples[index].Vars == nil {
							samples[index].Vars = make([][3]Generator, len(s), len(s))
						}
						samples[index].Vars[v][0] = x
						samples[index].Vars[v][1] = y
						samples[index].Vars[v][2] = z
						index++
					}
				}
			}
		}
	}

	o.Cost(samples, a...)
	sort.Slice(samples, func(i, j int) bool {
		return samples[i].Cost < samples[j].Cost
	})
	length := o.Length
	if o.Reg {
		panic("Reg currently not supported")
	}
	/*if o.Reg {
		length++
		x := NewMatrix(len(samples), 1)
		for _, s := range samples {
			x.Data = append(x.Data, float32(s.Cost))
		}
		sample := Sample{
			Cost: samples[0].Cost / 2,
			Vars: make([][3]Matrix, len(samples[0].Vars)),
		}
		for v := 0; v < len(o.Vars); v++ {
			for i := 0; i < 3; i++ {
				sample.Vars[v][i] = NewZeroMatrix(samples[0].Vars[v][i].Cols, samples[0].Vars[v][i].Rows)
				for j := 0; j < len(samples[0].Vars[v][i].Data); j++ {
					y := NewMatrix(len(samples), 1)
					for _, s := range samples {
						y.Data = append(y.Data, s.Vars[v][i].Data[j])
					}
					b0, b1 := LinearRegression(x, y)
					sample.Vars[v][i].Data[j] = float32(b1*sample.Cost + b0)
				}
			}
		}
		samples = append(samples, sample)
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].Cost < samples[j].Cost
		})
	}*/

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

	if stddev == 0 {
		for j := range o.Vars {
			for v := range o.Vars[j] {
				vv := NewRandomMatrix(o.Vars[j][v].Cols, o.Vars[j][v].Rows)
				for k := range vv.Data {
					vv.Data[k].StdDev = 0
				}
				for k := range samples {
					for l, value := range samples[k].Vars[j][v].Sample().Data {
						vv.Data[l].Mean += float64(value) / float64(o.Length)
					}
				}
				for k := range samples {
					for l, value := range samples[k].Vars[j][v].Sample().Data {
						diff := vv.Data[l].Mean - float64(value)
						vv.Data[l].StdDev += diff * diff / float64(o.Length)
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

	weights, sum := make([]float64, length, length), 0.0
	zero := samples[0].Cost
	up, down := 0, 0
	limit := 0
	upper, lower := mean+2*stddev, mean-3*stddev
	for i := range weights {
		if samples[i].Cost > upper {
			limit = i
			up++
			break
		} else if samples[i].Cost < lower {
			down++
		}
		diff := (samples[i].Cost - zero)
		weight := math.Exp(-(diff * float64(i)))
		sum += weight
		weights[i] = weight
	}
	//fmt.Println(up, down)
	for i := range weights {
		weights[i] /= sum
	}

	for j := range o.Vars {
		for v := range o.Vars[j] {
			vv := NewRandomMatrix(o.Vars[j][v].Cols, o.Vars[j][v].Rows)
			for k := range vv.Data {
				vv.Data[k].StdDev = 0
			}
			for k := range samples[:limit] {
				for l, value := range samples[k].Vars[j][v].Sample().Data {
					vv.Data[l].Mean += weights[k] * float64(value)
				}
			}
			for k := range samples[:limit] {
				for l, value := range samples[k].Vars[j][v].Sample().Data {
					diff := vv.Data[l].Mean - float64(value)
					vv.Data[l].StdDev += weights[k] * diff * diff
				}
			}
			for k := range vv.Data {
				vv.Data[k].StdDev /= (float64(o.Length) - 1.0) / float64(o.Length)
				vv.Data[k].StdDev = math.Sqrt(vv.Data[k].StdDev)
			}
			/*vvv := NewRandomMatrix(o.Vars[j][v].Cols, o.Vars[j][v].Rows)
			for k := range vv.Data {
				vvv.Data[k].StdDev = 0
			}
			d := samples[0].Vars[j][v].Distribution
			loss := make([]float64, d.Cols*d.Rows)
			for k := range samples {
				for l, value := range samples[k].Vars[j][v].Sample().Data {
					upper, lower := vv.Data[l].Mean+5*vv.Data[l].StdDev, vv.Data[l].Mean-5*vv.Data[l].StdDev
					item := weights[k] * float64(value)
					if item < upper && item > lower {
						vvv.Data[l].Mean += item
					} else {
						loss[l] += weights[k]
					}
				}
			}
			for k := range samples {
				for l, value := range samples[k].Vars[j][v].Sample().Data {
					upper, lower := vv.Data[l].Mean+5*vv.Data[l].StdDev, vv.Data[l].Mean-5*vv.Data[l].StdDev
					item := weights[k] * float64(value)
					if item < upper && item > lower {
						diff := vvv.Data[l].Mean - float64(value)
						vvv.Data[l].StdDev += weights[k] * diff * diff / ((1 - loss[l]) + 1e-9)
					}
				}
			}
			fmt.Println(loss)
			for k := range vvv.Data {
				vvv.Data[k].StdDev /= (float64(o.Length) - 1.0) / float64(o.Length)
				vvv.Data[k].StdDev = math.Sqrt(vvv.Data[k].StdDev)
			}*/
			o.Vars[j][v] = vv
		}
	}

	return samples[0]
}

// Optimize optimizes a cost function
func (o *Optimizer) Optimize(dx float64) Sample {
	last := -1.0
	for {
		s := o.Iterate()
		if last > 0 && math.Abs(last-s.Cost) < dx {
			return s
		}
		last = s.Cost
	}
}

// Meta is the meta optimizer
func Meta(metaSamples int, metaMin, metaScale float64, rng *Rand, n int, scale float64, vars int, reg bool,
	cost func(samples []Sample, a ...Matrix), a ...Matrix) Sample {
	source := make([][6]RandomMatrix, vars, vars)
	if vars != len(a) {
		for i := range source {
			for j := range source[i] {
				source[i][j] = NewRandomMatrix(a[0].Cols, a[0].Rows)
			}
		}
	} else {
		for i := range source {
			for j := range source[i] {
				source[i][j] = NewRandomMatrix(a[i].Cols, a[i].Rows)
			}
		}
	}
	type Meta struct {
		Optimizer
		Cost   float64
		Sample Sample
	}
	for {
		metas := make([]Meta, metaSamples)
		for i := range metas {
			metas[i].N = n
			metas[i].Length = n * n * n
			metas[i].Scale = scale
			seed := rng.Uint32() + 1
			if seed == 0 {
				seed = 1
			}
			rng := Rand(seed)
			metas[i].Rng = &rng
			metas[i].Vars = make([][3]RandomMatrix, vars)
			for j := range metas[i].Vars {
				for k := range metas[i].Vars[j] {
					dist := NewRandomMatrix(source[j][k].Cols, source[j][k].Rows)
					for l := range source[j][k].Data {
						r := source[j][k].Data[l]
						dist.Data[l].Mean = rng.NormFloat64()*r.StdDev + r.Mean
						r = source[j][k+3].Data[l]
						dist.Data[l].StdDev = rng.NormFloat64()*r.StdDev + r.Mean
					}
					metas[i].Vars[j][k] = dist
				}
			}
			metas[i].Optimizer.Cost = cost
			metas[i].Optimizer.Reg = reg
		}

		index, flight, cpus, done := 0, 0, runtime.NumCPU(), make(chan bool, 8)
		process := func(i int) {
			s := metas[i].Optimize(1e-6)
			metas[i].Cost = s.Cost
			metas[i].Sample = s
			done <- true
		}
		for index < len(metas) && flight < cpus {
			go process(index)
			index++
			flight++
		}
		for index < len(metas) {
			<-done
			flight--

			go process(index)
			index++
			flight++
		}
		for i := 0; i < flight; i++ {
			<-done
		}

		sort.Slice(metas, func(i, j int) bool {
			return metas[i].Cost < metas[j].Cost
		})
		if metas[0].Cost < metaMin {
			return metas[0].Sample
		}

		mean, stddev := 0.0, 0.0
		for i := range metas {
			mean += metas[i].Cost
		}
		mean /= float64(len(metas))
		for i := range metas {
			diff := mean - metas[i].Cost
			stddev += diff * diff
		}
		stddev /= float64(len(metas))
		stddev = math.Sqrt(stddev)

		if stddev == 0 {
			for j := range source {
				for v := range source[j][:3] {
					vv := NewRandomMatrix(source[j][v].Cols, source[j][v].Rows)
					for k := range vv.Data {
						vv.Data[k].StdDev = 0
					}
					for k := range metas {
						for l, value := range metas[k].Vars[j][v].Data {
							vv.Data[l].Mean += float64(value.Mean) / float64(len(metas))
						}
					}
					for k := range metas {
						for l, value := range metas[k].Vars[j][v].Data {
							diff := vv.Data[l].Mean - float64(value.Mean)
							vv.Data[l].StdDev += diff * diff / float64(len(metas))
						}
					}
					for k := range vv.Data {
						vv.Data[k].StdDev /= (float64(len(metas)) - 1.0) / float64(len(metas))
						vv.Data[k].StdDev = math.Sqrt(vv.Data[k].StdDev)
					}
					source[j][v] = vv
				}
				devs := source[j][3:]
				for v := range devs {
					vv := NewRandomMatrix(devs[v].Cols, devs[v].Rows)
					for k := range vv.Data {
						vv.Data[k].StdDev = 0
					}
					for k := range metas {
						for l, value := range metas[k].Vars[j][v].Data {
							vv.Data[l].Mean += float64(value.StdDev) / float64(len(metas))
						}
					}
					for k := range metas {
						for l, value := range metas[k].Vars[j][v].Data {
							diff := vv.Data[l].Mean - float64(value.StdDev)
							vv.Data[l].StdDev += diff * diff / float64(len(metas))
						}
					}
					for k := range vv.Data {
						vv.Data[k].StdDev /= (float64(len(metas)) - 1.0) / float64(len(metas))
						vv.Data[k].StdDev = math.Sqrt(vv.Data[k].StdDev)
					}
					devs[v] = vv
				}
			}

			continue
		}

		weights, sum := make([]float64, len(metas), len(metas)), 0.0
		for i := range weights {
			diff := (metas[i].Cost - mean) / stddev
			weight := math.Exp(-(diff*diff/2 + metaScale*float64(i))) / (stddev * math.Sqrt(2*math.Pi))
			sum += weight
			weights[i] = weight
		}
		for i := range weights {
			weights[i] /= sum
		}

		for j := range source {
			for v := range source[j][:3] {
				vv := NewRandomMatrix(source[j][v].Cols, source[j][v].Rows)
				for k := range vv.Data {
					vv.Data[k].StdDev = 0
				}
				for k := range metas {
					for l, value := range metas[k].Vars[j][v].Data {
						vv.Data[l].Mean += weights[k] * float64(value.Mean)
					}
				}
				for k := range metas {
					for l, value := range metas[k].Vars[j][v].Data {
						diff := vv.Data[l].Mean - float64(value.Mean)
						vv.Data[l].StdDev += weights[k] * diff * diff
					}
				}
				for k := range vv.Data {
					vv.Data[k].StdDev /= (float64(len(metas)) - 1.0) / float64(len(metas))
					vv.Data[k].StdDev = math.Sqrt(vv.Data[k].StdDev)
				}
				source[j][v] = vv
			}
			devs := source[j][3:]
			for v := range devs {
				vv := NewRandomMatrix(devs[v].Cols, devs[v].Rows)
				for k := range vv.Data {
					vv.Data[k].StdDev = 0
				}
				for k := range metas {
					for l, value := range metas[k].Vars[j][v].Data {
						vv.Data[l].Mean += weights[k] * float64(value.StdDev)
					}
				}
				for k := range metas {
					for l, value := range metas[k].Vars[j][v].Data {
						diff := vv.Data[l].Mean - float64(value.StdDev)
						vv.Data[l].StdDev += weights[k] * diff * diff
					}
				}
				for k := range vv.Data {
					vv.Data[k].StdDev /= (float64(len(metas)) - 1.0) / float64(len(metas))
					vv.Data[k].StdDev = math.Sqrt(vv.Data[k].StdDev)
				}
				devs[v] = vv
			}
		}
	}
}
