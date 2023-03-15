package linear_models

// #include <stdlib.h>
import "C"

import (
	"errors"
	"fmt"
	"github.com/azurmof/golearn/base"
	"strconv"
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

	weightClasses := base.GetClassDistribution(X)
	weightVec := make([]float64, len(weightClasses))
	classLabels := make([]int32, len(weightClasses))

	idx := 0
	for classLabel, _ := range weightClasses {
		weightVec[idx] = 1.0
		intClassLabel, err := strconv.Atoi(classLabel)
		if err != nil {
			return err
		}
		classLabels[idx] = int32(intClassLabel)
		idx++
	}

	// Allocate memory for cWeightLabel and cWeight using C.malloc
	lr.param.cWeightLabel = (*C.int)(C.malloc(C.size_t(len(weightClasses)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(lr.param.cWeightLabel))

	lr.param.cWeight = (*C.double)(C.malloc(C.size_t(len(weightVec)) * C.size_t(unsafe.Sizeof(C.double(0)))))
	defer C.free(unsafe.Pointer(lr.param.cWeight))

	// Copy the values from classLabels and weightVec to cWeightLabel and cWeight
	for idx, v := range classLabels {
		*(*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(lr.param.cWeightLabel)) + uintptr(idx)*unsafe.Sizeof(C.int(0)))) = C.int(v)
	}

	for idx, v := range weightVec {
		*(*C.double)(unsafe.Pointer(uintptr(unsafe.Pointer(lr.param.cWeight)) + uintptr(idx)*unsafe.Sizeof(C.double(0)))) = C.double(v)
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
