//go:build !noasm && amd64
// AUTO-GENERATED BY GOAT -- DO NOT EDIT

package vector

import "unsafe"

//go:noescape
func _mm256_dot(a, b, n, ret unsafe.Pointer)
