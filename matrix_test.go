// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"math"
	"math/rand"
	"testing"
)

func TestNorm(t *testing.T) {
	rng := Rand(1)
	a := NewMatrix(2, 2)
	a.Data = append(a.Data,
		3, 8,
		4, 6,
	)
	identity := NewIdentityMatrix(a.Cols)
	optimizer := NewOptimizer(&rng, 8, .1, 1, func(samples []Sample, x ...Matrix) {
		done := make(chan bool, 8)
		process := func(index int) {
			x := samples[index].Vars[0][0].Sample()
			y := samples[index].Vars[0][1].Sample()
			z := samples[index].Vars[0][2].Sample()
			ai := x.Add(y.H(z))
			cost := a.MulT(ai).Quadratic(identity).Avg()
			samples[index].Cost = float64(cost.Data[0])
			done <- true
		}
		for j := range samples {
			go process(j)
		}
		for range samples {
			<-done
		}
	}, a)
	optimizer.Norm = true
	s := optimizer.Optimize(1e-6)
	ai := s.Vars[0][0].Sample().Add(s.Vars[0][1].Sample().H(s.Vars[0][2].Sample()))
	b := a.MulT(ai)
	t.Log(b)
	for i := 0; i < b.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			value := float64(b.Data[i*b.Cols+j])
			if i == j {
				if math.Round(value) != 1 {
					t.Log(ai, b)
					t.Fatalf("result %d,%d should be 1 but is %f", i, j, value)
				}
			} else {
				if math.Round(value) != 0 {
					t.Log(ai, b)
					t.Fatalf("result %d,%d should be 0 but is %f", i, j, value)
				}
			}
		}
	}
}

func TestLU(t *testing.T) {
	a := NewMatrix(2, 2)
	a.Data = append(a.Data,
		3, 8,
		4, 6,
	)
	l, u := a.LU()
	b := l.Mul(u)
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
	l, u = a.LU()
	t.Log(l.MulT(u.T()))
}

func TestDeterminant(t *testing.T) {
	a := NewMatrix(2, 2)
	a.Data = append(a.Data,
		3, 8,
		4, 6,
	)
	d, err := a.Determinant()
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
	d, err = a.Determinant()
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
	d, err = a.Determinant()
	if err != nil {
		panic(err)
	}
	if d != 0 {
		t.Fatal("determinant should be 0", d)
	}
}

func TestInverse(t *testing.T) {
	rng := Rand(1)
	a := NewMatrix(2, 2)
	a.Data = append(a.Data,
		3, 8,
		4, 6,
	)
	ai := a.Inverse(&rng)
	b := a.MulT(ai)
	t.Log(b)
	for i := 0; i < b.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			value := float64(b.Data[i*b.Cols+j])
			if i == j {
				if math.Round(value) != 1 {
					t.Log(ai, b)
					t.Fatalf("result %d,%d should be 1 but is %f", i, j, value)
				}
			} else {
				if math.Round(value) != 0 {
					t.Log(ai, b)
					t.Fatalf("result %d,%d should be 0 but is %f", i, j, value)
				}
			}
		}
	}

	a = NewMatrix(3, 3)
	a.Data = append(a.Data,
		6, 1, 1,
		4, -2, 5,
		2, 8, 7,
	)
	ai = a.Inverse(&rng)
	b = a.MulT(ai)
	t.Log(b)
	for i := 0; i < b.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			value := float64(b.Data[i*b.Cols+j])
			if i == j {
				if math.Round(value) != 1 {
					t.Log(ai, b)
					t.Fatalf("result %d,%d should be 1 but is %f", i, j, value)
				}
			} else {
				if math.Round(value) != 0 {
					t.Log(ai, b)
					t.Fatalf("result %d,%d should be 0 but is %f", i, j, value)
				}
			}
		}
	}
}

func TestMulti(t *testing.T) {
	rng := Rand(1)
	for i := 0; i < 32; i++ {
		t.Log(i)
		multi := Multi{
			E: NewMatrix(2, 2),
			U: NewMatrix(2, 1),
		}
		multi.E.Data = append(multi.E.Data, 1, 3.0/5.0, 3.0/5.0, 2)
		multi.U.Data = append(multi.U.Data, 0, 0)
		multi.LearnA(&rng, nil)
		e := multi.A.MulT(multi.A.T())
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

func TestIris(t *testing.T) {
	const (
		// Inputs is the number of inputs
		Inputs = 4
		// Outputs is the number of outputs
		Outputs = 4
		// Embedding is the embedding size
		Embedding = 3 * 4
		// Clusters is the number of clusters
		Clusters = 3
	)
	type Flower struct {
		Measures  []float32
		Label     float32
		I         int
		Embedding []float32
		Cluster   int
	}

	rng := rand.New(rand.NewSource(1))
	flowers := make([]Flower, len(Iris))
	for i := range flowers {
		measures := make([]float32, 4)
		copy(measures, Iris[i][:4])
		flowers[i].Measures = measures
		flowers[i].Label = Iris[i][4]
	}

	for i, value := range flowers {
		sum := float32(0.0)
		for _, v := range value.Measures {
			sum += v * v
		}
		length := float32(math.Sqrt(float64(sum)))
		for i := range value.Measures {
			value.Measures[i] /= length
		}
		flowers[i].I = i
	}
	net := NewNet(4, Inputs, Outputs)
	length := len(flowers)
	const epochs = 3
	for i := 0; i < epochs; i++ {
		perm := rng.Perm(len(flowers))
		for epoch := 0; epoch < length; epoch++ {
			index := perm[epoch]
			query := NewMatrix(Inputs, 1, flowers[index].Measures...)
			key := NewMatrix(Inputs, 1, flowers[index].Measures...)
			value := NewMatrix(Inputs, 1, flowers[index].Measures...)
			label := flowers[index].Label
			entropy, q, k, v := net.Fire(query, key, value)
			t.Log(label, entropy, v.Data)
			if i == epochs-1 {
				flowers[index].Embedding = append(flowers[index].Embedding, q.Data...)
				flowers[index].Embedding = append(flowers[index].Embedding, k.Data...)
				flowers[index].Embedding = append(flowers[index].Embedding, v.Data...)
			}
		}
	}
	in := NewMatrix(Embedding, len(flowers))
	max := float32(0.0)
	for i := range flowers {
		for _, value := range flowers[i].Embedding {
			if value < 0 {
				value = -value
			}
			if value > max {
				max = value
			}
		}
	}
	for i := range flowers {
		for _, value := range flowers[i].Embedding {
			in.Data = append(in.Data, value/max)
		}
	}
	gmm := NewGMM(in, Clusters)
	out := gmm.Optimize(in)
	for i, value := range out {
		flowers[i].Cluster = value
	}
	ab, ba := [3][3]float64{}, [3][3]float64{}
	for i := range flowers {
		t.Log(IrisNames[flowers[i].Label], flowers[i].Cluster)
		a := int(flowers[i].Label)
		b := flowers[i].Cluster
		ab[a][b]++
		ba[b][a]++
	}
	entropy := 0.0
	for i := 0; i < 3; i++ {
		entropy += (1.0 / 3.0) * math.Log(1.0/3.0)
	}
	t.Log(-entropy, -(1.0/3.0)*math.Log(1.0/3.0))
	for i := range ab {
		entropy := 0.0
		for _, value := range ab[i] {
			if value > 0 {
				p := value / 150
				entropy += p * math.Log(p)
			}
		}
		entropy = -entropy
		t.Log("ab", i, entropy)
		if entropy > .6 {
			t.Fatal("entropy is greater than .5")
		}
	}
	for i := range ba {
		entropy := 0.0
		for _, value := range ba[i] {
			if value > 0 {
				p := value / 150
				entropy += p * math.Log(p)
			}
		}
		entropy = -entropy
		t.Log("ba", i, entropy)
		if entropy > .6 {
			t.Fatal("entropy is greater than .5")
		}
	}
}

func TestIrisSimplified(t *testing.T) {
	const (
		// Inputs is the number of inputs
		Inputs = 4
		// Outputs is the number of outputs
		Outputs = 4
		// Embedding is the embedding size
		Embedding = 3 * 4
		// Clusters is the number of clusters
		Clusters = 3
	)
	type Flower struct {
		Measures []float32
		Label    float32
		I        int
		Cluster  int
	}

	flowers := make([]Flower, len(Iris))
	for i := range flowers {
		measures := make([]float32, 4)
		copy(measures, Iris[i][:4])
		flowers[i].Measures = measures
		flowers[i].Label = Iris[i][4]
	}

	max := float32(0.0)
	for _, value := range flowers {
		for _, v := range value.Measures {
			if v > max {
				max = v
			}
		}
	}
	for i, value := range flowers {
		for i := range value.Measures {
			value.Measures[i] /= max
		}
		flowers[i].I = i
	}
	in := NewMatrix(4, len(flowers))
	for i := range flowers {
		in.Data = append(in.Data, flowers[i].Measures...)
	}
	out := MetaGMM(in, 3)
	for i, value := range out {
		flowers[i].Cluster = value
	}
	ab, ba := [3][3]float64{}, [3][3]float64{}
	for i := range flowers {
		t.Log(IrisNames[flowers[i].Label], flowers[i].Cluster)
		a := int(flowers[i].Label)
		b := flowers[i].Cluster
		ab[a][b]++
		ba[b][a]++
	}
	entropy := 0.0
	for i := 0; i < 3; i++ {
		entropy += (1.0 / 3.0) * math.Log(1.0/3.0)
	}
	t.Log(-entropy, -(1.0/3.0)*math.Log(1.0/3.0))
	for i := range ab {
		entropy := 0.0
		for _, value := range ab[i] {
			if value > 0 {
				p := value / 150
				entropy += p * math.Log(p)
			}
		}
		entropy = -entropy
		t.Log("ab", i, entropy)
		if entropy > .57 {
			t.Fatal("entropy is greater than .5")
		}
	}
	for i := range ba {
		entropy := 0.0
		for _, value := range ba[i] {
			if value > 0 {
				p := value / 150
				entropy += p * math.Log(p)
			}
		}
		entropy = -entropy
		t.Log("ba", i, entropy)
		if entropy > .57 {
			t.Fatal("entropy is greater than .5")
		}
	}
}

func TestLinearRegression(t *testing.T) {
	x := NewMatrix(10, 1, []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}...)
	y := NewMatrix(10, 1, []float32{1, 3, 2, 5, 7, 8, 8, 9, 10, 12}...)
	b0, b1 := LinearRegression(x, y)
	if math.Round(float64(b0)*1000) != math.Round(1.236*1000) {
		t.Fatal("result should be 1.236", b0)
	}
	if math.Round(float64(b1)*1000) != math.Round(1.170*1000) {
		t.Fatal("result should be 1.170", b1)
	}
}

func BenchmarkGoRand(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rng.NormFloat64()
	}
}

func BenchmarkCustomRand(b *testing.B) {
	rng := Rand(1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rng.NormFloat64()
	}
}

func BenchmarkMulti(b *testing.B) {
	rng := Rand(1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		multi := Multi{
			E: NewMatrix(2, 2),
			U: NewMatrix(2, 1),
		}
		multi.E.Data = append(multi.E.Data, 1, 3.0/5.0, 3.0/5.0, 2)
		multi.U.Data = append(multi.U.Data, 0, 0)
		multi.LearnA(&rng, nil)
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

func BenchmarkInverse(b *testing.B) {
	rng := Rand(1)
	a := NewMatrix(2, 2)
	a.Data = append(a.Data,
		3, 8,
		4, 6,
	)
	identity := NewIdentityMatrix(a.Cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer := NewOptimizer(&rng, 10, .1, 1, func(samples []Sample, x ...Matrix) {
			done := make(chan bool, 8)
			process := func(index int) {
				x := samples[index].Vars[0][0].Sample()
				y := samples[index].Vars[0][1].Sample()
				z := samples[index].Vars[0][2].Sample()
				cost := a.MulT(x.Add(y.H(z))).Quadratic(identity).Avg()
				samples[index].Cost = float64(cost.Data[0])
				done <- true
			}
			for j := range samples {
				go process(j)
			}
			for range samples {
				<-done
			}
		}, a)
		optimizer.Optimize(1e-9)
	}
}

func BenchmarkInverseMeta(b *testing.B) {
	rng := Rand(1)
	a := NewMatrix(2, 2)
	a.Data = append(a.Data,
		3, 8,
		4, 6,
	)
	identity := NewIdentityMatrix(a.Cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Meta(128, .1, .1, &rng, 4, .1, 1, false, func(samples []Sample, x ...Matrix) {
			done := make(chan bool, 8)
			process := func(index int) {
				x := samples[index].Vars[0][0].Sample()
				y := samples[index].Vars[0][1].Sample()
				z := samples[index].Vars[0][2].Sample()
				cost := a.MulT(x.Add(y.H(z))).Quadratic(identity).Avg()
				samples[index].Cost = float64(cost.Data[0])
				done <- true
			}
			for j := range samples {
				go process(j)
			}
			for range samples {
				<-done
			}
		}, a)
	}
}
