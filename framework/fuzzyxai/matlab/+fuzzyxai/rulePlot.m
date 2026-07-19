function figureHandle = rulePlot(result)
%RULEPLOT Plot at most seven evidence-backed primary rules.
if isstring(result) || ischar(result), result = fuzzyxai.loadResult(string(result)); end
fuzzyxai.validateResult(result);
rules = result.layers.rules; count = min(numel(rules), 7);
values = zeros(1, count); labels = strings(1, count);
for index = 1:count
    labels(index) = string(rules(index).rule_id);
    if ~isempty(rules(index).importance), values(index) = rules(index).importance; end
end
figureHandle = figure("Color", "white"); barh(values); yticks(1:count); yticklabels(labels);
title("Primary rule importance; native/surrogate status is stored in JSON");
end
