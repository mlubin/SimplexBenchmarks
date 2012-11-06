
cd("MATLAB")
output = readall(`matlab -nodisplay -r "runbench('../$(ARGS[1])');quit"`)
(s,finish) = search(output,"Problem is")
print(output[s:end])
cd("..")
