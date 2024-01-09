// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package experimental

import (
	"math"
	"math/rand"
	"testing"

	. "github.com/pointlander/matrix"
)

func TestLUFactor(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	a := NewMatrix(0, 2, 2)
	a.Data = append(a.Data,
		3, 8,
		4, 6,
	)
	l, u := LUFactor(rng, a)
	b := MulT(l, T(u))
	if math.Round(float64(b.Data[0])) != 3 {
		t.Fatal("result should be 3", b.Data[0])
	}
	if math.Round(float64(b.Data[1])) != 8 {
		t.Fatal("result should be 8", b.Data[1])
	}
	if math.Round(float64(b.Data[2])) != 4 {
		t.Fatal("result should be 4", b.Data[2])
	}
	if math.Round(float64(b.Data[3])) != 6 {
		t.Fatal("result should be 6", b.Data[3])
	}
}
