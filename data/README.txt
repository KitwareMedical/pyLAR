LRA (low-rank atlas-to-image registration) framework is introduced in  our
MICCAI 2014 paper: "Low-Rank to the Rescue - Atlas-Based Analyses in the
Presence of Pathologies".

We starated with a greedy implementation, in which each iteration updates newly
deformed input images from the previous iteration results. The later non-greedy
implemntation can avoid potential accumulated errors by updating the total
deformation at each iteration from the original input images (use the
deformation from the previous iteration to initialize).

Three datasets are used for experiments:1) Bull's eye dataset (concentric
spheres with varing grayscal intensities simulated using python scripts;2)
MICCAI'12 BRATS data: both synthetic and patient brain tumor MRI images;3)
2D female and male face images.
