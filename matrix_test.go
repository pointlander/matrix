// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"math"
	"math/rand"
	"testing"
)

func TestDeterminant(t *testing.T) {
	a := NewMatrix(0, 2, 2)
	a.Data = append(a.Data,
		3, 8,
		4, 6,
	)
	d, err := Determinant(a)
	if err != nil {
		panic(err)
	}
	if d != -14 {
		t.Fatal("determinant should be -14")
	}

	a = NewMatrix(0, 3, 3)
	a.Data = append(a.Data,
		6, 1, 1,
		4, -2, 5,
		2, 8, 7,
	)
	d, err = Determinant(a)
	if err != nil {
		panic(err)
	}
	if d != -306 {
		t.Fatal("determinant should be -306")
	}

	a = NewMatrix(0, 4, 4)
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
		t.Fatal("determinant should be 0")
	}
}

func TestMulti(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 32; i++ {
		t.Log(i)
		multi := Multi{
			E: NewMatrix(1, 2, 2),
			U: NewMatrix(1, 2, 1),
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
			E: NewMatrix(1, 2, 2),
			U: NewMatrix(1, 2, 1),
		}
		multi.E.Data = append(multi.E.Data, 1, 3.0/5.0, 3.0/5.0, 2)
		multi.U.Data = append(multi.U.Data, 0, 0)
		multi.LearnA(rng, nil)
	}
}

func TestMultiTest(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 32; i++ {
		t.Log(i)
		multi := Multi{
			E: NewMatrix(1, 2, 2),
			U: NewMatrix(1, 2, 1),
		}
		multi.E.Data = append(multi.E.Data, 1, 3.0/5.0, 3.0/5.0, 2)
		multi.U.Data = append(multi.U.Data, 0, 0)
		multi.LearnAWithRandomSearch(rng, nil)
		e := MulT(multi.A, T(multi.A))
		if math.Round(float64(e.Data[0])*10) != 10 {
			t.Fatal("result should be 1", e.Data[0])
		}
		if math.Round(float64(e.Data[1])*10) != 6 {
			t.Fatal("result should be 6", e.Data[1])
		}
		if math.Round(float64(e.Data[2])*10) != 6 {
			t.Fatal("result should be 6", e.Data[2])
		}
		if math.Round(float64(e.Data[3])*10) != 20 {
			t.Fatal("result should be 2", e.Data[3])
		}
	}
}

func BenchmarkMultiTest(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		multi := Multi{
			E: NewMatrix(1, 2, 2),
			U: NewMatrix(1, 2, 1),
		}
		multi.E.Data = append(multi.E.Data, 1, 3.0/5.0, 3.0/5.0, 2)
		multi.U.Data = append(multi.U.Data, 0, 0)
		multi.LearnAWithRandomSearch(rng, nil)
	}
}
