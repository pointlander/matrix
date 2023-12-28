// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"math"
	"testing"
)

func TestMulti(t *testing.T) {
	multi := Multi{
		E: NewMatrix(1, 2, 2),
		U: NewMatrix(1, 2, 1),
	}
	multi.E.Data = append(multi.E.Data, 1, 3.0/5.0, 3.0/5.0, 2)
	multi.U.Data = append(multi.U.Data, 0, 0)
	multi.LearnA(nil)
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

func TestMultiTest(t *testing.T) {
	multi := Multi{
		E: NewMatrix(1, 2, 2),
		U: NewMatrix(1, 2, 1),
	}
	multi.E.Data = append(multi.E.Data, 1, 3.0/5.0, 3.0/5.0, 2)
	multi.U.Data = append(multi.U.Data, 0, 0)
	multi.LearnATest(nil)
	e := MulT(multi.A, T(multi.A))
	t.Log(multi.A)
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
