// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"math"
	"math/rand"
	"testing"
)

func TestLU(t *testing.T) {
	a := NewMatrix(2, 2)
	a.Data = append(a.Data,
		3, 8,
		4, 6,
	)
	l, u := LU(a)
	b := Mul(l, u)
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

	a = NewMatrix(3, 3)
	a.Data = append(a.Data,
		6, 1, 1,
		4, -2, 5,
		2, 8, 7,
	)
	l, u = LU(a)
	t.Log(MulT(l, T(u)))
}

func TestDeterminant(t *testing.T) {
	a := NewMatrix(2, 2)
	a.Data = append(a.Data,
		3, 8,
		4, 6,
	)
	d, err := Determinant(a)
	if err != nil {
		panic(err)
	}
	if math.Round(float64(d)) != -14 {
		t.Fatal("determinant should be -14", d)
	}

	a = NewMatrix(3, 3)
	a.Data = append(a.Data,
		6, 1, 1,
		4, -2, 5,
		2, 8, 7,
	)
	d, err = Determinant(a)
	if err != nil {
		panic(err)
	}
	if math.Round(float64(d)) != -306 {
		t.Fatal("determinant should be -306", d)
	}

	a = NewMatrix(4, 4)
	a.Data = append(a.Data,
		1, 3, 1, 2,
		5, 8, 5, 3,
		0, 4, 0, 0,
		2, 3, 2, 8,
	)
	d, err = Determinant(a)
	if err != nil {
		panic(err)
	}
	if d != 0 {
		t.Fatal("determinant should be 0", d)
	}
}

func TestInverse(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	a := NewMatrix(2, 2)
	a.Data = append(a.Data,
		3, 8,
		4, 6,
	)
	ai := Inverse(rng, a)
	b := MulT(a, ai)
	if math.Round(float64(b.Data[0])) != 1 {
		t.Fatal("result should be 1", b.Data[0])
	}
	if math.Round(float64(b.Data[1])) != 0 {
		t.Fatal("result should be 0", b.Data[1])
	}
	if math.Round(float64(b.Data[2])) != 0 {
		t.Fatal("result should be 0", b.Data[2])
	}
	if math.Round(float64(b.Data[3])) != 1 {
		t.Fatal("result should be 1", b.Data[3])
	}
}

func TestMulti(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 32; i++ {
		t.Log(i)
		multi := Multi{
			E: NewMatrix(2, 2),
			U: NewMatrix(2, 1),
		}
		multi.E.Data = append(multi.E.Data, 1, 3.0/5.0, 3.0/5.0, 2)
		multi.U.Data = append(multi.U.Data, 0, 0)
		multi.LearnA(rng, nil)
		e := MulT(multi.A, T(multi.A))
		if math.Round(float64(e.Data[0])*10) != 10 {
			t.Fatal("result should be 1")
		}
		if math.Round(float64(e.Data[1])*10) != 6 {
			t.Fatal("result should be 6")
		}
		if math.Round(float64(e.Data[2])*10) != 6 {
			t.Fatal("result should be 6")
		}
		if math.Round(float64(e.Data[3])*10) != 20 {
			t.Fatal("result should be 2")
		}
	}
}

func BenchmarkMulti(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		multi := Multi{
			E: NewMatrix(2, 2),
			U: NewMatrix(2, 1),
		}
		multi.E.Data = append(multi.E.Data, 1, 3.0/5.0, 3.0/5.0, 2)
		multi.U.Data = append(multi.U.Data, 0, 0)
		multi.LearnA(rng, nil)
	}
}

func BenchmarkFullAllocate(b *testing.B) {
	const length = 1024 * 1024
	for i := 0; i < b.N; i++ {
		data := make([]int, length, length)
		for i := range data {
			data[i] = i
		}
	}
}

func BenchmarkTruncatedAllocate(b *testing.B) {
	const length = 1024 * 1024
	for i := 0; i < b.N; i++ {
		data := make([]int, 0, length)
		for i := 0; i < length; i++ {
			data = append(data, i)
		}
	}
}
