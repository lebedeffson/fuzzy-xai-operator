function figureHandle = explanationStory(result)
%EXPLANATIONSTORY Render Data -> Training -> Knowledge -> Decision -> Action.
if ischar(result)
    result = fuzzyxai.loadResult(result);
end
fuzzyxai.validateResult(result);
figureHandle = figure("Name", "FuzzyXAI explanation story", "Color", "white");
axis([0 1 0 1]); axis off; hold on;
title("Claim-centered explanation story");
if ~isfield(result, "visual_spec") || ~isfield(result.visual_spec, "story")
    text(0.05, 0.75, "VisualSpec unavailable; regenerate the explanation with FuzzyXAI >= 1.1.");
    hold off; return;
end
stages = result.visual_spec.story;
if iscell(stages)
    count = numel(stages);
else
    count = numel(stages);
end
boxWidth = 0.16; gap = 0.035;
for index = 1:count
    if iscell(stages), stage = stages{index}; else, stage = stages(index); end
    x = 0.025 + (index - 1) * (boxWidth + gap);
    color = localStatusColor(char(stage.status));
    rectangle("Position", [x 0.30 boxWidth 0.42], "EdgeColor", color, "LineWidth", 2, "Curvature", 0.05);
    text(x + 0.01, 0.67, char(stage.title), "FontWeight", "bold", "Interpreter", "none");
    text(x + 0.01, 0.62, upper(char(stage.status)), "Color", color, "Interpreter", "none");
    facts = stage.facts;
    if iscell(facts), factCount = numel(facts); else, factCount = numel(facts); end
    for factIndex = 1:min(factCount, 3)
        if iscell(facts), fact = facts{factIndex}; else, fact = facts(factIndex); end
        text(x + 0.01, 0.56 - 0.09 * (factIndex - 1), char(fact), "FontSize", 8, "Interpreter", "none");
    end
    if index < count
        annotation("arrow", [x + boxWidth, x + boxWidth + gap], [0.51, 0.51]);
    end
end
text(0.025, 0.16, "Every displayed statement is linked to an ExplanationClaim and evidence reference.", "Color", [0.2 0.36 0.45]);
hold off;
end

function color = localStatusColor(status)
if strcmp(status, "supported")
    color = [0.18 0.42 0.31];
elseif strcmp(status, "conflict")
    color = [0.64 0.23 0.23];
elseif strcmp(status, "limitation")
    color = [0.72 0.47 0.12];
else
    color = [0.48 0.53 0.56];
end
end
