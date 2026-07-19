function figureHandle = trainingTrajectory(result, objectIndex)
%TRAININGTRAJECTORY Plot confidence/loss and observed forgetting events.
if nargin < 2, objectIndex = 1; end
if isstring(result) || ischar(result), result = fuzzyxai.loadResult(string(result)); end
trace = result.layers.training(objectIndex);
figureHandle = figure("Color", "white");
plot(trace.confidence_by_epoch, "-o", "DisplayName", "confidence"); hold on;
plot(trace.loss_by_epoch, "-s", "DisplayName", "loss"); hold off;
legend("Location", "best"); xlabel("Epoch"); title("Object " + string(trace.object_id) + " training trajectory");
end
