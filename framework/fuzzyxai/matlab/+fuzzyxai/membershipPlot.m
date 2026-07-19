function figureHandle = membershipPlot(result)
%MEMBERSHIPPLOT Plot fuzzy memberships supplied by the operator pipeline.
if isstring(result) || ischar(result), result = fuzzyxai.loadResult(string(result)); end
memberships = result.fuzzy.memberships; names = fieldnames(memberships);
values = cellfun(@(name) memberships.(name), names);
figureHandle = figure("Color", "white"); bar(values); ylim([0 1]);
xticks(1:numel(names)); xticklabels(names); title("Fuzzy memberships for the explained object");
end
