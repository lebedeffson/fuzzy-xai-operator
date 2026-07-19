function figureHandle = similarCasesPlot(result)
%SIMILARCASESPLOT Plot scores with explicit similarity method labels.
if isstring(result) || ischar(result), result = fuzzyxai.loadResult(string(result)); end
cases = result.layers.similar_cases; count = numel(cases);
values = zeros(1, count); labels = strings(1, count);
for index = 1:count
    values(index) = cases(index).similarity_score;
    labels(index) = string(cases(index).reference_object_id) + " / " + string(cases(index).similarity_method);
end
figureHandle = figure("Color", "white"); barh(values); xlim([0 1]);
yticks(1:count); yticklabels(labels); title("Similar cases; metric and compared representation are explicit in JSON");
end
