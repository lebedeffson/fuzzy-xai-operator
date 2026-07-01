from __future__ import annotations

from typing import Mapping


def _grid(n: int = 101) -> list[float]:
    if n <= 1:
        return [0.0]
    return [i / (n - 1) for i in range(n)]


def _polish(fig, *, title: str, xaxis_title: str, yaxis_title: str, height: int = 360):
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor='left', font=dict(size=17)),
        font=dict(family='Arial, sans-serif', size=12, color='#16202a'),
        xaxis=dict(
            title=xaxis_title,
            showgrid=True,
            gridcolor='#edf1f6',
            zeroline=False,
            linecolor='#d9dee7',
        ),
        yaxis=dict(
            title=yaxis_title,
            range=[0, 1.05],
            showgrid=True,
            gridcolor='#edf1f6',
            zeroline=False,
            linecolor='#d9dee7',
        ),
        height=height,
        margin=dict(l=45, r=18, t=58, b=42),
        legend=dict(orientation='h', yanchor='bottom', y=-0.28, x=0, xanchor='left'),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        hovermode='x unified',
    )
    return fig


def representation_figure(obj, *, title: str = 'A_k^F representation', n: int = 101):
    """Return a Plotly figure for F0, IntervalFS, HesitantFS, NeutrosophicFS, or MultiLevelFS.

    The function is intentionally dependency-light: Plotly is imported lazily so the
    scientific core remains usable without UI packages.
    """
    try:
        import plotly.graph_objects as go
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('plotly is required for representation figures') from exc

    xs = _grid(n)
    fig = go.Figure()
    cls_name = getattr(obj, 'class_name', type(obj).__name__)

    if cls_name in {'F0', 'base'} or hasattr(obj, 'mu'):
        ys = [float(obj.membership(x)) for x in xs]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='degree', line=dict(width=3, color='#2563eb')))

    elif cls_name == 'FI' or hasattr(obj, 'lower'):
        lo = [float(obj.membership(x)[0]) for x in xs]
        hi = [float(obj.membership(x)[1]) for x in xs]
        fig.add_trace(go.Scatter(x=xs, y=hi, mode='lines', name='upper bound', line=dict(width=3, color='#2563eb')))
        fig.add_trace(go.Scatter(x=xs, y=lo, mode='lines', name='lower bound', fill='tonexty', fillcolor='rgba(37,99,235,0.14)', line=dict(width=3, color='#7aa2f7')))

    elif cls_name == 'FH' or hasattr(obj, 'values_fn'):
        x_vals = []
        y_vals = []
        for x in xs[::5]:
            for value in obj.membership(x):
                x_vals.append(x)
                y_vals.append(value)
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='expert values', marker=dict(size=9, color='#c47b00', line=dict(color='white', width=1))))

    elif cls_name in {'FNsrc', 'FN'} or hasattr(obj, 't_fn'):
        # show the center point and an optional trajectory over the support
        x0 = 0.72
        t, i, f = obj.membership(x0)
        fig.add_trace(go.Bar(
            x=['T support', 'I indeterminacy', 'F rejection'],
            y=[t, i, f],
            name=f'x0={x0}',
            marker_color=['#0f9f6e', '#c47b00', '#d83a3a'],
            text=[round(t, 3), round(i, 3), round(f, 3)],
            textposition='outside',
        ))

    elif cls_name == 'FML' or hasattr(obj, 'levels'):
        for idx, level in enumerate(obj.levels):
            try:
                level_fig = representation_figure(level, title=f'level {idx+1}', n=n)
                for trace in level_fig.data:
                    trace.name = f'L{idx+1}: {trace.name}'
                    fig.add_trace(trace)
            except Exception:
                continue

    else:
        fig.add_annotation(text=f'No visualization adapter for {type(obj).__name__}', x=0.5, y=0.5, showarrow=False)

    _polish(fig, title=title, xaxis_title='normalized value', yaxis_title='degree', height=360)
    return fig


def explainplan_membership_figure(plan, feature_name: str | None, *, df=None, n: int = 121):
    """Robust ExplainPlan figure for numeric and categorical features.

    Always returns a valid figure with either curves or an explicit error/info note.
    """
    import plotly.graph_objects as go
    colors = {'low': '#0f9f6e', 'medium': '#c47b00', 'high': '#d83a3a'}

    def _error_figure(message: str):
        fig = go.Figure()
        fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
        _polish(fig, title='Membership visualization', xaxis_title='x', yaxis_title='degree', height=360)
        return fig

    try:
        metadata = getattr(plan, 'metadata', {}) or {}
        feature_terms = metadata.get('feature_terms', {}) or {}
        numeric_features = list(metadata.get('numeric_features', []) or [])
        feature = feature_name or (numeric_features[0] if numeric_features else None)

        # Default fallback: canonical low/medium/high on [0,1].
        if not feature or feature not in feature_terms:
            xs = _grid(n)
            low = [max(0.0, (0.5 - x) / 0.5) for x in xs]
            medium = [4*x-1 if 0.25 <= x <= 0.5 else (3-4*x if 0.5 < x <= 0.75 else 0.0) for x in xs]
            high = [4*x-2 if 0.5 <= x <= 0.75 else (1.0 if x > 0.75 else 0.0) for x in xs]
            fig = go.Figure()
            for name, ys in [('low', low), ('medium', medium), ('high', high)]:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode='lines', name=name,
                    fill='tozeroy',
                    fillcolor={'low': 'rgba(15,159,110,0.10)', 'medium': 'rgba(196,123,0,0.10)', 'high': 'rgba(216,58,58,0.10)'}[name],
                    line=dict(width=4, color=colors[name], shape='spline'),
                ))
            _polish(fig, title='Функции принадлежности', xaxis_title='normalized value', yaxis_title='degree', height=360)
            return fig

        spec = feature_terms.get(feature, {})
        ftype = spec.get('type', 'numeric')

        # Categorical feature: show a frequency bar chart (more understandable than empty axes).
        if ftype == 'categorical':
            values = list(spec.get('values', []) or [])
            counts: Mapping[str, int] = {}
            if df is not None and feature in getattr(df, 'columns', []):
                value_counts = df[feature].astype(str).value_counts().to_dict()
                counts = {str(k): int(v) for k, v in value_counts.items()}
            else:
                counts = {str(v): 1 for v in values}
            xs = list(counts.keys())[:20]
            ys = [counts[x] for x in xs]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=xs, y=ys, name='count', marker_color='#2563eb', text=ys, textposition='outside'))
            fig.add_annotation(
                text='Для категориальных признаков показывается частота категорий.<br>Нечёткие кривые строятся для числовых признаков.',
                x=0.5, y=1.08, xref='paper', yref='paper', showarrow=False,
            )
            _polish(fig, title=f'Категориальный признак: {feature}', xaxis_title=feature, yaxis_title='count', height=360)
            fig.update_yaxes(range=None)
            return fig

        # Numeric feature with term specs.
        q = spec.get('quantiles', {}) or {}
        lo = float(q.get('min', 0.0))
        hi = float(q.get('max', 1.0))
        if hi <= lo:
            hi = lo + 1.0
        xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        terms = spec.get('terms', {}) or {}

        if not terms:
            return _error_figure(f'Для признака "{feature}" не найдены термы')

        def _mu(term: Mapping[str, float], x: float) -> float:
            kind = str(term.get('kind', 'triangle'))
            if kind == 'left_shoulder':
                a = float(term.get('a', lo)); b = float(term.get('b', hi))
                if x <= a:
                    return 1.0
                if x >= b:
                    return 0.0
                return (b - x) / max(b - a, 1e-12)
            if kind == 'right_shoulder':
                a = float(term.get('a', lo)); b = float(term.get('b', hi))
                if x <= a:
                    return 0.0
                if x >= b:
                    return 1.0
                return (x - a) / max(b - a, 1e-12)
            a = float(term.get('a', lo)); b = float(term.get('b', (lo + hi) / 2.0)); c = float(term.get('c', hi))
            if x <= a or x >= c:
                return 0.0
            if abs(x - b) < 1e-12:
                return 1.0
            if x < b:
                return (x - a) / max(b - a, 1e-12)
            return (c - x) / max(c - b, 1e-12)

        fig = go.Figure()
        for term_name, term_spec in terms.items():
            ys = [_mu(term_spec, x) for x in xs]
            name = str(term_name)
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode='lines',
                name=name,
                fill='tozeroy',
                fillcolor={
                    'low': 'rgba(15,159,110,0.10)',
                    'medium': 'rgba(196,123,0,0.10)',
                    'high': 'rgba(216,58,58,0.10)',
                }.get(name, 'rgba(37,99,235,0.10)'),
                line=dict(width=4, color=colors.get(name, '#2563eb'), shape='spline'),
                hovertemplate=f'{name}<br>{feature}=%{{x:.3f}}<br>degree=%{{y:.3f}}<extra></extra>',
            ))
        for key in ('q25', 'median', 'q75'):
            if key in q:
                fig.add_vline(x=float(q[key]), line_width=1, line_dash='dot', line_color='#94a3b8')
        _polish(fig, title=f'Функции принадлежности: {feature}', xaxis_title=feature, yaxis_title='degree', height=360)
        return fig
    except Exception as exc:
        return _error_figure(f'Ошибка построения графика: {exc}')
