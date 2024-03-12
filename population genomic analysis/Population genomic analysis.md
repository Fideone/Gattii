# Population genomic analysis

```SHELL
#population genomic analysis

for i in sample.list

do

bwa index -a bwtsw FungiDB-63_CgattiiVGIIR265_Genome.fasta
samtools faidx FungiDB-63_CgattiiVGIIR265_Genome.fasta
picard CreateSequenceDictionary R=FungiDB-63_CgattiiVGIIR265_Genome.fasta O=FungiDB-63_CgattiiVGIIR265_Genome.dict

bwa mem -t 40 -M -Y -R '@RG\tID:'${i}'WGS\tSM:'${i}'\tLB:WGS\tPL:Illumina' FungiDB-63_CgattiiVGIIR265_Genome.fasta ${i}/${i}_1.rd.fastq ${i}/${i}_2.rd.fastq > ${i}.sam
samtools view -bS ${i}.sam  -o ${i}.bam 
samtools view -h -b -q30  ${i}.bam -o  ${i}.q30.bam 
gatk SortSam -I ${i}.q30.bam -O ${i}.q30.sort.bam  -SO coordinate
gatk MarkDuplicates -I ${i}.q30.sort.bam -O ${i}.q30.sort.markdup.bam -M ${i}.q30.sort.markdup_metrics.txt
samtools index ${i}.q30.sort.markdup.bam

  gatk HaplotypeCaller -native-pair-hmm-threads 16 -R FungiDB-63_CgattiiVGIIR265_Genome.fasta  -ERC GVCF -ploidy 1  -I ${i}.q30.sort.markdup.bam -O ${i}.sorted.bam.g.vcf
done

gatk CombineGVCFs  -R FungiDB-63_CgattiiVGIIR265_Genome.fasta   --variant  sample.list  -O H99129.vcf


gatk   GenotypeGVCFs -R FungiDB-63_CgattiiVGIIR265_Genome.fasta -V  H99129.vcf -O H99129geno.vcf   


gatk  SelectVariants -R FungiDB-63_CgattiiVGIIR265_Genome.fasta -V H99129geno.vcf      -O  H99129_snp.vcf --select-type-to-include SNP  --restrict-alleles-to BIALLELIC 


gatk VariantFiltration  -R FungiDB-63_CgattiiVGIIR265_Genome.fasta  \
   -V  H99129_snp.vcf \
   -O H99129_snp.filter.vcf  --filter-expression "QUAL < 30.0 || QD < 2.0 || SOR > 3.0 || MQ < 40.0 || FS > 60.0" --filter-name "Filter"  
   
gatk SelectVariants -V  H99129_snp.filter.vcf  --exclude-filtered true -O H99129_snp_filtered.vcf  


python3 vcf2phylip.py -i H99129_snp_filtered.vcf

modeltest-ng -i H99129_snp_filtered.min4.phy  -d nt -p 40  

raxml-ng --all  --msa H99129_snp_filtered.min4.phy --model GTR

vcftools --vcf H99129_snp_filtered.vcf --recode --recode-INFO-all --stdout  --remove-indv  CNH99 > R265128_snp_filtered.vcf

gatk CountVariants -V  R265128_snp_filtered.vcf  

plink --vcf R265128_snp_filtered.vcf    --recode   --allow-extra-chr   --out R265128PCA_result 

plink --allow-extra-chr \
  --file R265128PCA_result \
  --out 128PCA_filtered_snps  \ 
  --pca 10
  
plink --file R265128PCA_result  --indep-pairwise 50 10 0.2 --allow-extra-chr  --out R265128LDfiltered

plink  \
   --vcf R265128_snp_filtered.vcf \
   --extract R265128LDfiltered.prune.in \
   --out R265128LDfiltered.prune.in \
   --recode vcf-iid  \
   --allow-extra-chr 
   
plink   \
   --vcf R265128LDfiltered.prune.in.vcf  \
   --recode 12   \
   --out R265128LDfiltered.prune.in   \
   --allow-extra-chr
   
   
for K in $(seq 5 15); do admixture --cv R265128LDfiltered.prune.in.ped $K -j40 | tee log${K}.out; done

grep -h CV log*.out |sort -nk4  -t ' ' > cross-validation_error.txt

#PopLDdecay
cd LDdecay

PopLDdecay   -InVCF R265128LDfiltered.prune.in.vcf -OutStat  VGI.stat.gz  -SubPop    VGI.list

PopLDdecay   -InVCF R265128LDfiltered.prune.in.vcf -OutStat  VGII.stat.gz  -SubPop    VGII.list

ls *.stat.gz 

#make multi.list
VGI.stat.gz  VGI
VGII.stat.gz VGII

 perl  Plot_MultiPop.pl  -inList  multi.list  -output FigVGIVGII
 
 #π
cd π

#VGI
plink --vcf VGI53vcfpca_snp_filtered.vcf --maf 0.05 --geno 0.2 --recode vcf-iid -out VGI53vcfpcaWithId-maf0.05 --allow-extra-chr

vcftools --vcf  VGI53vcfpcaWithId-maf0.05.vcf --window-pi 500000  --window-pi-step 50000 --out VGI_pi

#VGII
plink --vcf VGII69VCFPCA_snp_filtered.vcf --maf 0.05 --geno 0.2 --recode vcf-iid -out VGII69VCFPCAWithId-maf0.05 --allow-extra-chr

#pi
vcftools --vcf VGII69VCFPCAWithId-maf0.05.vcf --window-pi 500000 --window-pi-step 50000  --out VGII_pi
```

