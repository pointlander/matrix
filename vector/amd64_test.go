// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && amd64
// +build !noasm,amd64

package vector

import (
	"math/rand"
	"testing"
)

const Size = 32 * 1024

func TestDot(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	x := make([]float32, Size)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	y := make([]float32, Size)
	for i := range y {
		y[i] = float32(rng.NormFloat64())
	}
	correct := goDot(x, y)
	if a := Dot(x, y); int(a*100) != int(correct*100) {
		t.Fatalf("dot product is broken %f != %f", a, correct)
	}
}

func BenchmarkVectorDot(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	x := make([]float32, Size)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	y := make([]float32, Size)
	for i := range y {
		y[i] = float32(rng.NormFloat64())
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Dot(x, y)
	}
}

func BenchmarkVectorBdot(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	x := make([]float32, Size)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	y := make([]float32, Size)
	for i := range y {
		y[i] = float32(rng.NormFloat64())
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Bdot(x, y)
	}
}

func goDot(x, y []float32) (z float32) {
	for i := range x {
		z += x[i] * y[i]
	}
	return z
}

func BenchmarkDot(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	x := make([]float32, Size)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	y := make([]float32, Size)
	for i := range y {
		y[i] = float32(rng.NormFloat64())
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		goDot(x, y)
	}
}
