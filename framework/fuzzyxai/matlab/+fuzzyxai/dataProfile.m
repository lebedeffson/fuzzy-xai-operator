function figureHandle = dataProfile(result)
%DATAPROFILE Render object percentiles against the reference interval.
if ischar(result), result = fuzzyxai.loadResult(result); end
fuzzyxai.validateResult(result);
figureHandle = figure("Name", "FuzzyXAI data profile", "Color", "white");
if ~isfield(result, "visual_spec") || ~isfield(result.visual_spec, "data_profile") || isempty(result.visual_spec.data_profile)
    text(0.5, 0.5, "Reference profile unavailable", "HorizontalAlignment", "center"); axis off; return;
end
profiles = result.visual_spec.data_profile;
count = numel(profiles); values = zeros(1, count); labels = cell(1, count);
for index = 1:count
    values(index) = profiles(index).percentile;
    labels{index} = char(profiles(index).feature);
end
barh(values, "FaceColor", [0.18 0.42 0.31]); xlim([0 100]);
yticks(1:count); yticklabels(labels); xlabel("Percentile against reference");
title("Object position; not an automatic data-error verdict");
end
