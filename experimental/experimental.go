// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package experimental

import (
	. "github.com/pointlander/matrix"
)

// LUFactor factors a matrix into lower and upper
func LUFactor(rng *Rand, a Matrix) (l, u Matrix) {
	s := Meta(128, .1, .1, rng, 4, .1, 2, false, func(samples []Sample, x ...Matrix) {
		done := make(chan bool, 8)
		process := func(index int) {
			xl := samples[index].Vars[0][0].Sample()
			yl := samples[index].Vars[0][1].Sample()
			zl := samples[index].Vars[0][2].Sample()
			sl := xl.Add(yl.H(zl))
			xu := samples[index].Vars[1][0].Sample()
			yu := samples[index].Vars[1][1].Sample()
			zu := samples[index].Vars[1][2].Sample()
			su := xu.Add(yu.H(zu))
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
			samples[index].Cost = float64(sl.MulT(su.T()).Quadratic(a).Avg().Data[0])
			done <- true
		}
		for j := range samples {
			go process(j)
		}
		for range samples {
			<-done
		}
	}, a)
	xl := s.Vars[0][0].Sample()
	yl := s.Vars[0][1].Sample()
	zl := s.Vars[0][2].Sample()
	sl := xl.Add(yl.H(zl))
	xu := s.Vars[1][0].Sample()
	yu := s.Vars[1][1].Sample()
	zu := s.Vars[1][2].Sample()
	su := xu.Add(yu.H(zu))
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
	return sl, su
}
