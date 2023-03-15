package linear_models

// #include <stdlib.h>
import "C"

import (
	"errors"
	"fmt"
	"github.com/azurmof/golearn/base"
	"unsafe"
)

type LogisticRegression struct {
	param   *Parameter
	model   *Model
	problem *Problem
}

func NewLogisticRegression(penalty string, C float64, eps float64) (*LogisticRegression, error) {
	solver_type := 0
	if penalty == "l2" {
		solver_type = L2R_LR
	} else if penalty == "l1" {
		solver_type = L1R_LR
	} else {
		return nil, errors.New(fmt.Sprintf("Invalid penalty '%s'", penalty))
	}

	lr := LogisticRegression{}
	lr.param = NewParameter(solver_type, C, eps)
	lr.model = nil
	return &lr, nil
}

func (lr *LogisticRegression) Fit(X base.FixedDataGrid) error {
	problemVec := convertInstancesToProblemVec(X)
	labelVec := convertInstancesToLabelVec(X)
	lr.problem = NewProblem(problemVec, labelVec, 0)

	weightClasses := make([]int32, len(lr.param.WeightLabel))
	for i, v := range lr.param.WeightLabel {
		weightClasses[i] = int32(v)
	}

	// Allocate memory for cWeightLabel and cWeight using C.malloc
	lr.param.cWeightLabel = (*C.int)(C.malloc(C.size_t(len(weightClasses)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(lr.param.cWeightLabel))

	lr.param.cWeight = (*C.double)(C.malloc(C.size_t(len(lr.param.Weight)) * C.size_t(unsafe.Sizeof(C.double(0)))))
	defer C.free(unsafe.Pointer(lr.param.cWeight))

	// Copy the values from weightClasses and lr.param.Weight to cWeightLabel and cWeight
	for i, v := range weightClasses {
		*(*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(lr.param.cWeightLabel)) + uintptr(i)*unsafe.Sizeof(C.int(0)))) = C.int(v)
	}

	for i, v := range lr.param.Weight {
		*(*C.double)(unsafe.Pointer(uintptr(unsafe.Pointer(lr.param.cWeight)) + uintptr(i)*unsafe.Sizeof(C.double(0)))) = C.double(v)
	}

	// Train
	lr.model = Train(lr.problem, lr.param)
	return nil
}

func (lr *LogisticRegression) Predict(X base.FixedDataGrid) (base.FixedDataGrid, error) {

	// Only support 1 class Attribute
	classAttrs := X.AllClassAttributes()
	if len(classAttrs) != 1 {
		panic(fmt.Sprintf("%d Wrong number of classes", len(classAttrs)))
	}
	// Generate return structure
	ret := base.GeneratePredictionVector(X)
	classAttrSpecs := base.ResolveAttributes(ret, classAttrs)
	// Retrieve numeric non-class Attributes
	numericAttrs := base.NonClassFloatAttributes(X)
	numericAttrSpecs := base.ResolveAttributes(X, numericAttrs)

	// Allocate row storage
	row := make([]float64, len(numericAttrSpecs))
	X.MapOverRows(numericAttrSpecs, func(rowBytes [][]byte, rowNo int) (bool, error) {
		for i, r := range rowBytes {
			row[i] = base.UnpackBytesToFloat(r)
		}
		val := Predict(lr.model, row)
		vals := base.PackFloatToBytes(val)
		ret.Set(classAttrSpecs[0], rowNo, vals)
		return true, nil
	})

	return ret, nil
}

func (lr *LogisticRegression) String() string {
	return "LogisticRegression"
}
