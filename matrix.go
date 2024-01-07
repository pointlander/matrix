// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/matrix/vector"
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
	// Window is the window size
	Window = 64
	// Samples is the number of samples to take
	Samples = 1024
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Random is a random variable
type Random struct {
	Mean   float32
	StdDev float32
}

// RandomMatrix is a random matrix
type RandomMatrix struct {
	Cols int
	Rows int
	Data []Random
}

// NewRandomMatrix returns a new random matrix
func NewRandomMatrix(cols, rows int) RandomMatrix {
	data := make([]Random, cols*rows)
	factor := float32(math.Sqrt(2.0 / float64(cols)))
	for i := range data {
		data[i].StdDev = factor
	}
	return RandomMatrix{
		Cols: cols,
		Rows: rows,
		Data: data,
	}
}

// Sample samples a matrix
func (r RandomMatrix) Sample(rng *rand.Rand) Matrix {
	sample := NewMatrix(0, r.Cols, r.Rows)
	for _, v := range r.Data {
		sample.Data = append(sample.Data, float32(rng.NormFloat64())*v.StdDev+v.Mean)
	}
	return sample
}

// Matrix is a float32 matrix
type Matrix struct {
	Cols   int
	Rows   int
	Data   []float32
	States [][]float32
}

// NewMatrix32 creates a new float32 matrix
func NewMatrix(states, cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]float32, 0, cols*rows),
	}
	if states > 0 {
		m.States = make([][]float32, states)
		for i := range m.States {
			m.States[i] = make([]float32, cols*rows)
		}
	}
	return m
}

// Size is the size of the float32 matrix
func (m Matrix) Size() int {
	return m.Cols * m.Rows
}

// String returns a string for the matrix
func (m Matrix) String() string {
	s := "\n"
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			s += fmt.Sprintf(" %f", m.Data[i*m.Cols+j])
		}
		s += "\n"
	}
	return s + "\n"
}

// MulT multiplies two matrices and computes the transpose
func MulT(m Matrix, n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float32, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, vector.Dot(mm, nn))
		}
	}
	return o
}

// Add adds two float32 matrices
func Add(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// Sub subtracts two float32 matrices
func Sub(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value-n.Data[i%lenb])
	}
	return o
}

// Sigmoid computes the sigmoid of a matrix
func Sigmoid(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, float32(1/(1+math.Exp(-float64(value)))))
	}
	return o
}

// Step computes the step function of a float32 matrix
func Step(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		if value > 0 {
			value = 1
		} else {
			value = -1
		}
		o.Data = append(o.Data, value)
	}
	return o
}

// Quadratic computes the quadratic loss of two matrices
func Quadratic(m Matrix, n Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: 1,
		Data: make([]float32, 0, 1*m.Rows),
	}
	for i := 0; i < size; i += width {
		a, b, sum := m.Data[i:i+width], n.Data[i:i+width], float32(0.0)
		for j, ax := range a {
			diff := ax - b[j]
			sum += diff * diff
		}
		o.Data = append(o.Data, float32(math.Sqrt(float64(sum))))
	}
	return o
}

// QuadraticSet computes the quadratic loss of sub sets of two matrices
func QuadraticSet(m Matrix, n Matrix, set []int) Matrix {
	o := Matrix{
		Cols: 1,
		Rows: 1,
		Data: make([]float32, 0, 1),
	}
	sum := float32(0.0)
	for _, i := range set {
		diff := m.Data[i] - n.Data[i]
		sum += diff * diff
	}
	o.Data = append(o.Data, float32(math.Sqrt(float64(sum))))
	return o
}

// Avg computes the avg of a matrix
func Avg(m Matrix) Matrix {
	o := Matrix{
		Cols: 1,
		Rows: 1,
		Data: make([]float32, 0, 1),
	}
	sum := float32(0.0)
	for _, value := range m.Data {
		sum += value
	}
	o.Data = append(o.Data, sum/float32(len(m.Data)))
	return o
}

// T tramsposes a matrix
func T(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

// Normalize normalizes a matrix to the unit vector
func Normalize(m Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for i := 0; i < size; i += width {
		sum := float32(0.0)
		for _, ax := range m.Data[i : i+width] {
			sum += ax * ax
		}
		length := float32(math.Sqrt(float64(sum)))
		if sum == 0 {
			length = 1
		}
		for _, ax := range m.Data[i : i+width] {
			o.Data = append(o.Data, ax/length)
		}
	}
	return o
}

func softmax(values []float32) {
	max := float32(0.0)
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := float32(0.0)
	for j, value := range values {
		values[j] = float32(math.Exp(float64(value - s)))
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// SelfAttention computes the self attention of Q, K, V
func SelfAttention(Q, K, V Matrix) Matrix {
	o := Matrix{
		Cols: V.Rows,
		Rows: K.Rows,
		Data: make([]float32, 0, V.Rows*K.Rows),
	}
	outputs, values := make([]float32, V.Cols), make([]float32, Q.Rows)
	V = T(V)
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = vector.Dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			outputs[j] = vector.Dot(values, V)
		}
		softmax(outputs)
		o.Data = append(o.Data, outputs...)
	}
	return o
}

// SelfEntropy computes the self entropy of Q, K, V
func SelfEntropy(Q, K, V Matrix) ([]Matrix, []float32) {
	entropies, values, results := make([]float32, V.Cols), make([]float32, K.Rows), make([]float32, 0, K.Rows)
	outputs := make([]Matrix, 0, 8)
	V = T(V)
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = vector.Dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			entropies[j] = vector.Dot(values, V)
		}
		output := NewMatrix(0, V.Cols, 1)
		output.Data = append(output.Data, entropies...)
		outputs = append(outputs, output)
		softmax(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += float64(e) * math.Log(float64(e))
		}
		results = append(results, float32(-entropy))
	}
	return outputs, results
}

// Everett is the everett activation function
func Everett(m Matrix) Matrix {
	o := Matrix{
		Cols: 2 * m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, 2*m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		min, max := value, value
		if min > 0 {
			min = 0
		}
		if max < 0 {
			max = 0
		}
		o.Data = append(o.Data, min, max)
	}
	return o
}

// TaylorSoftmax is the taylor softmax
// https://arxiv.org/abs/1511.05042
func TaylorSoftmax(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	var sum float32
	columns, lenm := m.Cols, len(m.Data)
	for i := 0; i < lenm; i += columns {
		nn := m.Data[i : i+columns]
		for _, v := range nn {
			sum += 1 + v + v*v/2
		}
	}
	for i := 0; i < lenm; i += columns {
		nn := m.Data[i : i+columns]
		for _, v := range nn {
			o.Data = append(o.Data, (1+v+v*v/2)/sum)
		}
	}
	return o
}

// LU factors a matrix into lower and upper
func LU(rng *rand.Rand, a Matrix) (l, u Matrix) {
	window := 4
	mean, stddev := float32(0), float32(0)
	for _, value := range a.Data {
		mean += value
	}
	mean /= float32(len(a.Data))
	for _, value := range a.Data {
		diff := mean - value
		stddev += diff * diff
	}
	stddev = float32(math.Sqrt(float64(stddev)))
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
		Cost float32
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
				samples[j].Cost = cost.Data[0]
			}
			sort.Slice(samples, func(i, j int) bool {
				return samples[i].Cost < samples[j].Cost
			})

			weights, sum := make([]float32, window), float32(0)
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
						ll.Data[j].Mean += weights[i] * value
					}
				}
				for i := range samples[:window] {
					for j, value := range samples[i].L.Data {
						diff := ll.Data[j].Mean - value
						ll.Data[j].StdDev += weights[i] * diff * diff
					}
				}
				for i := range ll.Data {
					ll.Data[i].StdDev /= (float32(window) - 1.0) / float32(window)
					ll.Data[i].StdDev = float32(math.Sqrt(float64(ll.Data[i].StdDev)))
				}
				rl = ll
			} else {
				uu := NewRandomMatrix(a.Cols, a.Rows)
				for j := range uu.Data {
					uu.Data[j].StdDev = 0
				}
				for i := range samples[:window] {
					for j, value := range samples[i].U.Data {
						uu.Data[j].Mean += weights[i] * value
					}
				}
				for i := range samples[:window] {
					for j, value := range samples[i].U.Data {
						diff := uu.Data[j].Mean - value
						uu.Data[j].StdDev += weights[i] * diff * diff
					}
				}
				for i := range uu.Data {
					uu.Data[i].StdDev /= (float32(window) - 1.0) / float32(window)
					uu.Data[i].StdDev = float32(math.Sqrt(float64(uu.Data[i].StdDev)))
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

// https://d-caponi1.medium.com/matrix-determinants-in-go-b96aa3bcdc37
type stack []float32

func (s *stack) isEmpty() bool {
	return len(*s) == 0
}
func (s *stack) push(n float32) {
	*s = append(*s, n)
}
func (s *stack) pop() (float32, bool) {
	if s.isEmpty() {
		return 0, false
	}
	i := len(*s) - 1
	n := (*s)[i]
	*s = (*s)[:i]
	return n, true
}
func (s *stack) ToSlice() []float32 {
	return *s
}

func subMat(mat [][]float32, p int) [][]float32 {
	stacks := make([]stack, len(mat))
	for n := range mat {
		stacks[n] = stack{}
		for j := range mat[n] {
			if j != p {
				stacks[n].push(mat[n][j])
			}
		}
	}
	out := make([][]float32, len(mat))
	for k := range stacks {
		out[k] = stacks[k].ToSlice()
	}
	return out
}

// Determinant calculates the determinant of a matrix
func Determinant(a Matrix) (float32, error) {
	rng := rand.New(rand.NewSource(1))
	l, u := LU(rng, a)
	det := float32(1)
	for i := 0; i < l.Cols; i++ {
		det *= l.Data[i*l.Cols+i] * u.Data[i*l.Cols+i]
	}
	return det, nil
}

// Det calculates the determinant of a matrix
func Det(mat [][]float32) (float32, error) {
	// Base cases and rules
	if len(mat) != len(mat[0]) {
		return 0.0, errors.New("determinant can only be performed on square matrices")
	}
	if len(mat) == 1 {
		return (mat[0][0]), nil
	}
	if len(mat) == 2 {
		return (mat[0][0] * mat[1][1]) - (mat[0][1] * mat[1][0]), nil
	}
	s := float32(0.0) // accumulator
	for i := 0; i < len(mat[0]); i++ {

		sm := subMat(mat[1:][:], i) // peel off top row before passing
		z, err := Det(sm)           // get determinant of sub-matrix

		if err == nil {
			if i%2 != 0 {
				s -= mat[0][i] * z
			} else {
				s += mat[0][i] * z
			}
		}
	}
	return s, nil
}

// Multi is a multivariate distribution
type Multi struct {
	E Matrix
	U Matrix
	A Matrix
}

// NewMulti make a new multi
func NewMulti(vars int) Multi {
	factor := float32(math.Sqrt(2.0 / float64(vars)))
	a := NewMatrix(0, vars, vars)
	a.Data = a.Data[:cap(a.Data)]
	for i := 0; i < vars; i++ {
		for j := 0; j < vars; j++ {
			if i == j {
				a.Data[i*vars+j] = factor
			}
		}
	}
	u := NewMatrix(1, vars, 1)
	u.Data = u.Data[:cap(u.Data)]
	return Multi{
		A: a,
		U: u,
	}
}

// NewMultiFromData creates a new multivariate distribution
func NewMultiFromData(vars [][]float32) Multi {
	length := len(vars)
	e := NewMatrix(1, length, length)
	e.Data = e.Data[:cap(e.Data)]
	u := NewMatrix(1, length, 1)
	u.Data = u.Data[:cap(u.Data)]
	for i, v := range vars {
		for _, vv := range v {
			u.Data[i] += vv
		}
	}
	size := len(vars[0])
	for i := range u.Data {
		u.Data[i] /= float32(size)
	}
	for i := 0; i < length; i++ {
		for j := i; j < length; j++ {
			for k := 0; k < size; k++ {
				e.Data[i*length+j] += (vars[i][k] - u.Data[i]) * (vars[j][k] - u.Data[j])
			}
			e.Data[i*length+j] /= float32(size)
		}
	}
	for i := 0; i < length; i++ {
		for j := i + 1; j < length; j++ {
			e.Data[j*length+i] = e.Data[i*length+j]
		}
	}
	return Multi{
		E: e,
		U: u,
	}
}

// LearnAWithRandomSearch factors a matrix into AA^T
func (m *Multi) LearnAWithRandomSearch(rng *rand.Rand, debug *[]float32) {
	length := m.U.Cols
	a := NewRandomMatrix(length, length)
	type Sample struct {
		Cost   float32
		Matrix Matrix
	}
	samples := make([]Sample, Samples)
	for i := 0; i < 1024; i++ {
		for j := range samples {
			sample := a.Sample(rng)
			cost := Avg(Quadratic(MulT(sample, T(sample)), m.E))
			samples[j].Matrix = sample
			samples[j].Cost = cost.Data[0]
		}
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].Cost < samples[j].Cost
		})

		// https://stats.stackexchange.com/questions/6534/how-do-i-calculate-a-weighted-standard-deviation-in-excel
		weights, sum := make([]float32, Window), float32(0)
		for i := range weights {
			sum += 1 / samples[i].Cost
			weights[i] = 1 / samples[i].Cost
		}
		for i := range weights {
			weights[i] /= sum
		}

		aa := NewRandomMatrix(length, length)
		for j := range aa.Data {
			aa.Data[j].StdDev = 0
		}
		for i := range samples[:Window] {
			for j, value := range samples[i].Matrix.Data {
				aa.Data[j].Mean += weights[i] * value
			}
		}
		for i := range samples[:Window] {
			for j, value := range samples[i].Matrix.Data {
				diff := aa.Data[j].Mean - value
				aa.Data[j].StdDev += weights[i] * diff * diff
			}
		}
		for i := range aa.Data {
			aa.Data[i].StdDev /= (Window - 1.0) / Window
			aa.Data[i].StdDev = float32(math.Sqrt(float64(aa.Data[i].StdDev)))
		}
		a = aa
		if samples[0].Cost < 1e-6 {
			break
		}
	}
	m.A = samples[0].Matrix
}

// LearnA factores a matrix into AA^T
func (m *Multi) LearnA(rng *rand.Rand, debug *[]float32) {
	length := m.U.Cols

	set := tf32.NewSet()
	set.Add("A", length, length)
	set.Add("E", length, length)
	set.Weights[1].X = append(set.Weights[1].X, m.E.Data...)

	for _, w := range set.Weights[:1] {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rng.NormFloat64()*factor))
		}
	}

	deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	cost := tf32.Avg(tf32.Quadratic(tf32.Mul(set.Get("A"), tf32.T(set.Get("A"))), set.Get("E")))
	alpha, eta, iterations := float32(.01), float32(.01), 8*2048
	i := 0
	for i < iterations {
		total := float32(0.0)
		set.Zero()

		total += tf32.Gradient(cost).X[0]
		sum := float32(0.0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := float32(math.Sqrt(float64(sum)))
		scaling := float32(1.0)
		if norm > 1 {
			scaling = 1 / norm
		}

		w := set.Weights[0]
		for k, d := range w.D {
			deltas[0][k] = alpha*deltas[0][k] - eta*d*scaling
			set.Weights[0].X[k] += deltas[0][k]
		}

		if debug != nil {
			*debug = append(*debug, total)
		}
		i++
		if total < 1e-6 {
			break
		}
	}

	a := NewMatrix(0, set.Weights[0].S[0], set.Weights[0].S[1])
	for _, v := range set.Weights[0].X {
		a.Data = append(a.Data, v)
	}
	m.A = a
}

// Sample samples from the multivariate distribution
func (m Multi) Sample(rng *rand.Rand) Matrix {
	length := m.U.Cols
	s := NewMatrix(0, length, 1)
	for i := 0; i < length; i++ {
		s.Data = append(s.Data, float32(rng.NormFloat64()))
	}
	return Add(MulT(m.A, s), m.U)
}
