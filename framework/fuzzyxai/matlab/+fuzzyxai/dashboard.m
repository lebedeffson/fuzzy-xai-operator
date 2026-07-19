function figureHandle = dashboard(result)
%DASHBOARD Render five evidence layers from ExplanationViewModel 2.0.
if ischar(result)
    result = fuzzyxai.loadResult(result);
end
fuzzyxai.validateResult(result);

figureHandle = figure("Name", "FuzzyXAI: evidence path from data to action", "Color", "white");
subplot(3, 2, 1);
if isfield(result.layers, "data") && ~isempty(result.layers.data)
    data = result.layers.data(1);
    names = fieldnames(data.outlier_scores);
    values = cellfun(@(name) localNumeric(data.outlier_scores.(name)), names);
    barh(values, "FaceColor", [0.18 0.44 0.53]);
    yticks(1:numel(names)); yticklabels(names);
    xlabel("Robust distance from median");
else
    localUnavailable("Data-quality evidence unavailable");
end
title("1. Data and quality");

subplot(3, 2, 2);
if isfield(result.layers, "training") && ~isempty(result.layers.training)
    trace = result.layers.training(1);
    plot(trace.confidence_by_epoch, "-o", "DisplayName", "confidence"); hold on;
    plot(trace.loss_by_epoch, "-s", "DisplayName", "loss"); hold off;
    xlabel("Epoch"); legend("Location", "best");
else
    localUnavailable("Training evidence unavailable; no forgetting claim");
end
title("2. How the model learned");

subplot(3, 2, 3);
if isfield(result.layers, "rules") && ~isempty(result.layers.rules)
    rules = result.layers.rules;
    count = min(numel(rules), 7);
    values = zeros(1, count); labels = cell(1, count);
    for index = 1:count
        labels{index} = char(rules(index).rule_id);
        values(index) = localNumeric(rules(index).importance);
    end
    barh(values, "FaceColor", [0.18 0.44 0.53]);
    yticks(1:count); yticklabels(labels);
else
    localUnavailable("No auditable rules or concepts supplied");
end
title("3. What the model learned");

subplot(3, 2, 4);
score = localField(result.model, "score", 0);
barh(score, "FaceColor", [0.09 0.42 0.53]); xlim([0 1]);
xlabel("Model score (not action confidence)");
title("4. Why this result was produced");

subplot(3, 2, 5);
gamma = localField(result.disagreement, "gamma", 0);
delta = localField(result.disagreement, "delta", 0);
rho = localField(result.risk, "rho", 0);
imagesc([gamma delta rho], [0 1]); colormap(gca, hot); colorbar;
xticks([1 2 3]); xticklabels({'gamma', 'Delta', 'rho'}); yticks([]);
title(["5. Trust and action: " char(result.risk.action)]);

subplot(3, 2, 6); axis off;
if isfield(result, "human_explanations") && isfield(result.human_explanations, "user")
    text(0, 1, char(result.human_explanations.user.summary), "VerticalAlignment", "top", "Interpreter", "none");
else
    text(0, 1, char(result.narrative), "VerticalAlignment", "top", "Interpreter", "none");
end
title("Plain-language explanation");
end

function value = localField(structure, fieldName, defaultValue)
if isfield(structure, fieldName) && ~isempty(structure.(fieldName))
    value = localNumeric(structure.(fieldName));
else
    value = defaultValue;
end
end

function value = localNumeric(value)
if isempty(value) || ~isnumeric(value)
    value = 0;
end
end

function localUnavailable(message)
text(0.5, 0.5, message, "HorizontalAlignment", "center", "Interpreter", "none");
axis off;
end
