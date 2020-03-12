import sys
import os
import numpy as np
from collections import defaultdict

def main():
	genotype_file = "test_data_masked.txt"
	interval = 16

	if len(sys.argv) > 1:
		genotype_file = sys.argv[1]
	if len(sys.argv) > 2:
		interval = sys.argv[2]

	g_masked = np.genfromtxt(os.path.join(sys.path[0], genotype_file))

	g_imputed = impute(g_masked)

	# Run Clark's Algorithm on chunks of genotypes in intervals of [interval] SNPs
	print("Running Clark's Algorithm...")
	haplotypes_list = []
	for i in range(0, len(g_imputed[0]), interval):
		print("\r{0:.2f}%".format(100*(i+1)/len(g_imputed[0])), end='')
		haplotypes_list.append(clarksAlgo(g_imputed[:, i:i+interval]))

	print("\nPhasing...")
	results = phase(g_imputed, haplotypes_list, interval)

	print("\nFormatting and writing results...")
	formatted_results = np.transpose(results)
	with open(os.path.join(sys.path[0], "test_data_sol.txt"), 'w+') as f:
	    for result in formatted_results:
	    	line = " ".join(str(x) for x in result.astype(int).tolist())
	    	f.write(line + "\n")

	print("Phasing complete.")

# imputes missing values in masked genotype data
def impute(g_masked):
	g_masked[np.isnan(g_masked)] = 3 

	## Calculate frequencies for alleles
	p0 = np.sum(g_masked == 0, axis=1)
	p2 = np.sum(g_masked == 2, axis=1)
	p0 = p0 / (p0 + p2)
	p2 = 1 - p0

	## Sample from binomial dist with above frequencies
	for i, row in enumerate(g_masked):
	    x = np.random.binomial(n=1, p=p0[i], size=sum(g_masked[i] == 3))
	    for j, num in enumerate(g_masked[i]):
	        pos = 0
	        if g_masked[i][j] == 3:
	            g_masked[i][j] = 0 if x[pos] else 2
	            pos += 1

	return np.transpose(g_masked)

# checks if the haplotype is compatible with the given genotype
def isCompatible(genotype, haplotype):
	zeroes_match = np.all(haplotype[np.argwhere(genotype == 0)] == 0)
	ones_match = np.all(haplotype[np.argwhere(genotype == 2)] == 1)
	count = np.sum(haplotype[np.argwhere(genotype == 0)] == 1)
	count += np.sum(haplotype[np.argwhere(genotype == 2)] == 0)
	return zeroes_match and ones_match, count

# given just a genotype, generate a pair of haplotypes that are compatible based on most common SNP
def findPair(genotype, genotypes):
	pair_idx = np.transpose(np.argwhere(genotype == 1))[0]
	num_uncertain = len(np.where(genotype == 1)[0])
	uncertain_genotypes = genotypes[:, pair_idx]
	filled_snps = np.zeros(num_uncertain)
	for i in range(num_uncertain):
		counts = np.bincount(uncertain_genotypes[:, i].astype(int))
		filled_snps[i] = 0 if len(counts) < 3 or counts[0] > counts[2] else 1
	comp_snps = np.where(filled_snps == 1, 0, 1)

	h1 = np.copy(genotype)
	h2 = np.copy(h1)

	h1[pair_idx] = filled_snps
	h2[pair_idx] = comp_snps


	h1 = np.where(h1 == 2, 1, h1)
	h2 = np.where(h2 == 2, 1, h2)

	return h1.astype(float), h2.astype(float)

# given genotype and haplotype, find complement haplotype to satisfy phasing
def findComplement(genotype, haplotype):
	h2 = np.copy(haplotype)
	h2[np.all([haplotype == 0, genotype == 1], axis=0)] = 1
	h2[np.all([haplotype == 1, genotype == 1], axis=0)] = 0
	return h2.astype(float)

# find initial haplotype list to begin Clark's Algorithm
def clarksInit(genotypes, haplotypes):
	num_samples, num_snps = genotypes.shape


	h = []
	g = np.copy(genotypes)

	for j in range(num_snps):
		counts = np.bincount(g[:, j].astype(int))

		h_j = 0 if len(counts) < 3 or counts[0] > counts[2] else 1
		h.append(h_j)
		if h_j == 0:
			mask = g[:, j] != 2
			g = g[mask, :]
		elif h_j == 1:
			mask = g[:, j] != 0
			g = g[mask, :]

	h = np.array(h).astype(float)
	haplotypes[h.tobytes()] = 1


	for g in genotypes:
		compatible, error_count = isCompatible(g, h)
		if compatible:
	  		h2 = findComplement(g, h)
	  		haplotypes[h2.tobytes()] = 1

# Implementation of Clark's Algorithm
def clarksAlgo(genotypes):
	# dict of haplotypes we have discovered so far
	# byte representation of haplotype array : number of times we've seen this
	haplotypes = defaultdict(int)

	clarksInit(genotypes, haplotypes)
	for g in genotypes:
		pairFound = False
		for h in haplotypes:
			h = np.frombuffer(h)
			compatible, error_count = isCompatible(g, h)

			if compatible:
				h2 = findComplement(g, h)
				haplotypes[h.tobytes()] += 1
				haplotypes[h2.tobytes()] += 1
				pairFound = True
				break

		if not pairFound:
			h_pair = findPair(g, genotypes)
			haplotypes[h_pair[0].tobytes()] += 1
			haplotypes[h_pair[1].tobytes()] += 1

	return haplotypes

# Given haplotype list from Clark's Algorithm, phase each genotype
def phase(genotypes, haplotypes_list, interval):
	phased = np.zeros(shape=(2*genotypes.shape[0], genotypes.shape[1]))
	for i in range(len(genotypes)):
		print("\r{0:.2f}%".format(100*(i+1)/len(genotypes)), end='')
		for j in range(0, len(genotypes[0]), interval):
			g_i = genotypes[i, j:j+interval]
			for h in haplotypes_list[int(j/interval)]:
				h = np.frombuffer(h)
				compatible, error_count = isCompatible(g_i, h)
				if compatible:
					phased[2*i, j:j+interval] = h
					h2 = findComplement(g_i, h)
					phased[2*i+1, j:j+interval] = h2
					break

	return phased


if __name__ == "__main__":
	main()
