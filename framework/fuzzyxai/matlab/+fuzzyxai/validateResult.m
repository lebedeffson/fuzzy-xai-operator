function valid = validateResult(result)
%VALIDATERESULT Validate required ExplanationViewModel 2.0 sections.
required = {"schema_version", "model", "route", "risk", "trace", "layers", "explanation_graph", "human_explanations"};
for index = 1:numel(required)
    if ~isfield(result, required{index})
        error("fuzzyxai:InvalidResult", ["Missing field: " required{index}]);
    end
end
if ~strcmp(char(result.schema_version), "2.0")
    error("fuzzyxai:InvalidResult", "ExplanationViewModel schema_version must be 2.0");
end
valid = true;
end
