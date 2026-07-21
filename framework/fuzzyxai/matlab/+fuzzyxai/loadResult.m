function result = loadResult(path)
%LOADRESULT Load a canonical FuzzyXAI ExplanationViewModel JSON file.
result = jsondecode(fileread(path));
fuzzyxai.validateResult(result);
end
