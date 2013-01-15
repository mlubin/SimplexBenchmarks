from array import array
from time import time
from copy import deepcopy
import sys

Basic = 1
AtLower = 2
AtUpper = 3

dualTol = 1e-7

class SparseMatrixCSC:
	def __init__(self,nrow,ncol,colptr,rowval,nzval):
		self.nrow = nrow
		self.ncol = ncol
		self.colptr = colptr
		self.rowval = rowval
		self.nzval = nzval

class InstanceData:
	def __init__(self,A,Atrans):
		self.A = A
		self.Atrans = Atrans

class IterationData:
	def __init__(self,valid,variableState,priceInput,reducedCosts,normalizedTableauRow):
		self.valid = valid
		self.variableState = variableState
		self.priceInput = priceInput
		self.reducedCosts = reducedCosts
		self.normalizedTableauRow = normalizedTableauRow

class IndexedVector:
	def __init__(self,densevec):
		n = len(densevec)
		self.elts = array('d',n*[0.])
		self.nzidx = array('l',n*[0])
		self.nnz = 0
		for i in xrange(n):
			if abs(densevec[i]) > 1e-50:
				self.elts[i] = densevec[i]
				self.nzidx[self.nnz] = i
				self.nnz += 1

def readMat(f):
	s1 = f.readline().strip().split()
	nrow,ncol,nnz = int(s1[0]),int(s1[1]),int(s1[2])
	colptr = [int(s)-1 for s in f.readline().strip().split()]
	rowval = [int(s)-1 for s in f.readline().strip().split()]
	nzval = [float(s) for s in f.readline().strip().split()]
	assert(len(colptr) == ncol+1)
	assert(colptr[ncol] == nnz)
	assert(len(rowval) == nnz)
	assert(len(nzval) == nnz)

	return SparseMatrixCSC(nrow,ncol,array('l',colptr),array('l',rowval),array('d',nzval))

def readInstance(f):
	A = readMat(f)
	Atrans = readMat(f)
	return InstanceData(A,Atrans)

def readIteration(f):
	variableState = array('i',[int(s) for s in f.readline().strip().split()])
	priceInput = array('d',[float(s) for s in f.readline().strip().split()])
	reducedCosts = array('d',[float(s) for s in f.readline().strip().split()])
	normalizedTableauRow = array('d',[float(s) for s in f.readline().strip().split()])
	valid = len(variableState) > 0 and len(priceInput) > 0
	return IterationData(valid,variableState,priceInput, reducedCosts, normalizedTableauRow)


def doPrice(instance,d):
	A = instance.A
	nrow,ncol = A.nrow,A.ncol
	output = array('d',(nrow+ncol)*[0.])

	rho = d.priceInput
	Arv = A.rowval
	Anz = A.nzval
	varstate = d.variableState

	t = time()

	for i in xrange(ncol):
		if (varstate[i] == Basic): continue
		val = 0.
		for k in xrange(A.colptr[i],A.colptr[i+1]):
			val += rho[Arv[k]]*Anz[k]
		output[i] = val
	
	for i in xrange(nrow):
		k = i+ncol
		if (varstate[i] == Basic): continue
		output[k] = -rho[i]
	
	return time() - t

def doPriceHypersparse(instance,d):
	A = instance.A
	Atrans = instance.Atrans
	nrow,ncol = A.nrow,A.ncol
	outputelts = array('d',(nrow+ncol)*[0.])
	outputnzidx = array('i',(nrow+ncol)*[0])
	outputnnz = 0
	rho = IndexedVector(d.priceInput)
	rhoelts = rho.elts
	rhoidx = rho.nzidx

	Atrv = Atrans.rowval
	Atnz = Atrans.nzval

	t = time()

	for k in xrange(rho.nnz):
		row = rhoidx[k]
		elt = rhoelts[row]
		for j in xrange(Atrans.colptr[row],Atrans.colptr[row+1]):
			idx = Atrv[j]
			val = outputielts[idx]
			if (val != 0.):
				val += elt*Atnz[j]
				outputelts[idx] = val
			else:
				outputelts[idx] = elt*Atnz[j]
				outputnzidx[outputnnz] = idx
				outputnnz += 1
			outputelts[row+ncol] = -elt
			outputnzidx[outputnnz] = row+col
			outputnnz += 1

	return time()-t

def doTwoPassRatioTest(instance,d):

	nrow,ncol = instance.A.nrow,instance.A.ncol
	candidates = array('i',ncol*[0])
	ncandidates = 0
	thetaMax = 1e25
	pivotTol = 1e-7

	redcost = d.reducedCosts
	varstate = d.variableState
	tabrow = d.normalizedTableauRow

	t = time()

	for i in xrange(ncol+nrow):
		thisState = varstate[i]
		if thisState == Basic: continue
		pivotElt = tabrow[i]
		if (thisState == AtLower and pivotElt > pivotTol) or (thisState == AtUpper and pivotElt < -pivotTol):
			ratio = 0.
			if (pivotElt < 0.):
				ratio = (redcost[i] - dualTol)/pivotElt
			else:
				ratio = (redcost[i] + dualTol)/pivotElt
			if (ratio < thetaMax):
				thetaMax = ratio
				candidates[ncandidates] = i
				ncandidates += 1

	enter = -1
	maxAlpha = 0.

	for k in xrange(ncandidates):
		i = candidates[k]
		ratio = redcost[i]/tabrow[i]
		if (ratio <= thetaMax):
			absalpha = abs(tabrow[i])
			if (absalpha > maxAlpha):
				maxAlpha = absalpha
				enter = i
	
	return time()-t

def doTwoPassRatioTestHypersparse(instance,d):

	nrow,ncol = instance.A.nrow,instance.A.ncol
	candidates = array('i',ncol*[0])
	ncandidates = 0
	thetaMax = 1e25
	pivotTol = 1e-7

	redcost = d.reducedCosts
	varstate = d.variableState
	tabrow = IndexedVector(d.normalizedTableauRow)
	tabrowelts = tabrow.elts
	tabrowidx = tabrow.nzidx

	t = time()

	for k in xrange(tabrow.nnz):
		i = tabrowidx[k]
		thisState = varstate[i]
		if thisState == Basic: continue
		pivotElt = tabrowelts[i]
		if (thisState == AtLower and pivotElt > pivotTol) or (thisState == AtUpper and pivotElt < -pivotTol):
			ratio = 0.
			if (pivotElt < 0.):
				ratio = (redcost[i] - dualTol)/pivotElt
			else:
				ratio = (redcost[i] + dualTol)/pivotElt
			if (ratio < thetaMax):
				thetaMax = ratio
				candidates[ncandidates] = i
				ncandidates += 1

	enter = -1
	maxAlpha = 0.

	for k in xrange(ncandidates):
		i = candidates[k]
		ratio = redcost[i]/tabrowelts[i]
		if (ratio <= thetaMax):
			absalpha = abs(tabrowelts[i])
			if (absalpha > maxAlpha):
				maxAlpha = absalpha
				enter = i
	
	return time()-t

def doUpdateDuals(instance,d):
	nrow,ncol = instance.A.nrow,instance.A.ncol
	
	redcost = d.reducedCosts
	varstate = d.variableState
	tabrow = deepcopy(d.normalizedTableauRow)

	stepsize = 1.

	t = time()

	for i in xrange(nrow+ncol):
		dnew = redcost[i] - stepsize*tabrow[i]

		if varstate[i] == AtLower:
			if dnew >= dualTol:
				redcost[i] = dnew
			else:
				delta = -dnew-dualTol
				redcost[i] = -dualTol
		elif varstate[i] == AtUpper:
			if dnew <= dualTol:
				redcost[i] = dnew
			else:
				data = -dnew+dualTol
				redcost[i] = dualTol
	
	return time()-t

def doUpdateDualsHypersparse(instance,d):
	nrow,ncol = instance.A.nrow,instance.A.ncol
	
	redcost = d.reducedCosts
	varstate = d.variableState
	tabrow = IndexedVector(d.normalizedTableauRow)
	tabrowelts = tabrow.elts
	tabrowidx = tabrow.nzidx

	stepsize = 1.

	t = time()

	for j in xrange(tabrow.nnz):
		i = tabrowidx[j]
		dnew = redcost[i] - stepsize*tabrowelts[i]

		if varstate[i] == AtLower:
			if dnew >= dualTol:
				redcost[i] = dnew
			else:
				redcost[i] = -dualTol
		elif varstate[i] == AtUpper:
			if dnew <= dualTol:
				redcost[i] = dnew
			else:
				redcost[i] = dualTol
	
	return time()-t



if __name__ == "__main__":
	f = open(sys.argv[1],'r')
	instance = readInstance(f)
	print "Problem is",instance.A.nrow,"by",instance.A.ncol,"with",len(instance.A.nzval),"nonzeros"
	benchmarks = [(doPrice,"Matrix-transpose-vector product with non-basic columns"),
			(doPrice,"Hyper-sparse matrix-transpose-vector product"),
			(doTwoPassRatioTest,"Two-pass dual ratio test"),
			(doTwoPassRatioTestHypersparse,"Hyper-sparse two-pass dual ratio test"),
			(doUpdateDuals,"Update dual iterate with cost shifting"),
			(doUpdateDualsHypersparse,"Hyper-sparse update dual iterate with cost shifting")
			]

	timings = len(benchmarks)*[0.]
	nruns = 0
	while True:
		dat = readIteration(f)
		if not dat.valid:
			break
		for i in xrange(len(benchmarks)):
			func,name = benchmarks[i]
			timings[i] += func(instance,dat)
		nruns += 1
	
	print nruns,"simulated iterations"
	for i in xrange(len(benchmarks)):
		print "%s: %f sec" % (benchmarks[i][1],timings[i])
	
		
