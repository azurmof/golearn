package linear_models

// #include <stdlib.h>
import "C"

import (
	"fmt"
	"github.com/azurmof/golearn/base"
	"strconv"
)

type LogisticRegression struct {
	param   *Parameter
	model   *Model
	problem *Problem
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

	lr.param.WeightLabel = classLabels
	lr.param.Weight = weightVec

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
