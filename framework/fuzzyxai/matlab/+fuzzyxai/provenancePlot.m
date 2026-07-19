function figureHandle = provenancePlot(result)
%PROVENANCEPLOT Show graph relations and trace identifiers.
if isstring(result) || ischar(result), result = fuzzyxai.loadResult(string(result)); end
fuzzyxai.validateResult(result);
edges = result.explanation_graph.edges; graphObject = digraph;
for index = 1:numel(edges)
    graphObject = addedge(graphObject, string(edges(index).source), string(edges(index).target));
end
figureHandle = figure("Color", "white"); plot(graphObject, "Layout", "layered");
title("Explanation provenance graph");
end
