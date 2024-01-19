// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package experimental

import (
	"math"
	"math/rand"
	"sort"

	. "github.com/pointlander/matrix"
)

// LUFactor factors a matrix into lower and upper
func LUFactor(rng *rand.Rand, a Matrix) (l, u Matrix) {
	window := 4
	mean, stddev := 0.0, 0.0
	for _, value := range a.Data {
		mean += float64(value)
	}
	mean /= float64(len(a.Data))
	for _, value := range a.Data {
		diff := mean - float64(value)
		stddev += diff * diff
	}
	stddev = math.Sqrt(stddev)
	rl, ru := NewRandomMatrix(a.Cols, a.Rows), NewRandomMatrix(a.Cols, a.Rows)
	for i := range rl.Data {
		rl.Data[i].Mean = mean
		rl.Data[i].StdDev = stddev
	}
	for i := range ru.Data {
		ru.Data[i].Mean = mean
		ru.Data[i].StdDev = stddev
	}
	type Sample struct {
		Cost float64
		L    Matrix
		U    Matrix
	}
	samples := make([]Sample, 256)
	for i := 0; i < 2*1024; i++ {
		set := rng.Perm(len(rl.Data))
		for s := 0; s < len(rl.Data); s += len(rl.Data) / 2 {
			for j := range samples {
				sl, su := rl.Sample(rng), ru.Sample(rng)
				for x := 0; x < sl.Cols; x++ {
					for y := 0; y < x; y++ {
						sl.Data[y*sl.Cols+x] = 0
					}
				}
				for x := 0; x < su.Cols; x++ {
					for y := x + 1; y < su.Rows; y++ {
						su.Data[y*su.Cols+x] = 0
					}
				}
				end := s + len(rl.Data)/2
				if end > len(rl.Data) {
					end = len(rl.Data)
				}
				cost := Avg(QuadraticSet(MulT(sl, T(su)), a, set[s:end]))
				samples[j].L = sl
				samples[j].U = su
				samples[j].Cost = float64(cost.Data[0])
			}
			sort.Slice(samples, func(i, j int) bool {
				return samples[i].Cost < samples[j].Cost
			})

			weights, sum := make([]float64, window), 0.0
			for i := range weights {
				sum += 1 / samples[i].Cost
				weights[i] = 1 / samples[i].Cost
			}
			for i := range weights {
				weights[i] /= sum
			}

			if i%2 == 0 {
				ll := NewRandomMatrix(a.Cols, a.Rows)
				for j := range ll.Data {
					ll.Data[j].StdDev = 0
				}
				for i := range samples[:window] {
					for j, value := range samples[i].L.Data {
						ll.Data[j].Mean += weights[i] * float64(value)
					}
				}
				for i := range samples[:window] {
					for j, value := range samples[i].L.Data {
						diff := ll.Data[j].Mean - float64(value)
						ll.Data[j].StdDev += weights[i] * diff * diff
					}
				}
				for i := range ll.Data {
					ll.Data[i].StdDev /= (float64(window) - 1.0) / float64(window)
					ll.Data[i].StdDev = math.Sqrt(ll.Data[i].StdDev)
				}
				rl = ll
			} else {
				uu := NewRandomMatrix(a.Cols, a.Rows)
				for j := range uu.Data {
					uu.Data[j].StdDev = 0
				}
				for i := range samples[:window] {
					for j, value := range samples[i].U.Data {
						uu.Data[j].Mean += weights[i] * float64(value)
					}
				}
				for i := range samples[:window] {
					for j, value := range samples[i].U.Data {
						diff := uu.Data[j].Mean - float64(value)
						uu.Data[j].StdDev += weights[i] * diff * diff
					}
				}
				for i := range uu.Data {
					uu.Data[i].StdDev /= (float64(window) - 1.0) / float64(window)
					uu.Data[i].StdDev = math.Sqrt(uu.Data[i].StdDev)
				}
				ru = uu
			}
		}
		if samples[0].Cost < 1e-18 {
			break
		}
	}
	return samples[0].L, samples[0].U
}
