from dataclasses import asdict
import json
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import streamlit as st
from engine import Model, mean_inference, export_frame

st.set_page_config(page_title='Casino Estadístico · Camilo Guevara', page_icon='🎲', layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}h1,h2,h3{font-family:'Space Grotesk',sans-serif!important}
.block-container{padding-top:2.1rem;max-width:1580px}header[data-testid="stHeader"]{background:transparent}
.hero{border-bottom:1px solid #26344a;padding:0 0 26px;margin-bottom:24px}.eyebrow{font-size:11px;letter-spacing:3px;color:#46e3ba;font-weight:700}.hero h1{font-size:clamp(32px,4vw,54px);letter-spacing:-2px;margin:8px 0}.hero p{color:#91a4bc;max-width:720px;font-size:15px}.badge{display:inline-block;border:1px solid #315247;border-radius:30px;padding:6px 12px;color:#46e3ba;font-size:11px;letter-spacing:1px}.card{background:linear-gradient(145deg,#142237,#101827);border:1px solid #28374f;padding:18px 20px;border-radius:14px;margin:0 0 10px}.card label{font-size:10px;color:#91a4bc;letter-spacing:1.6px}.card .number{font-family:'Space Grotesk';font-size:31px;color:#edf5ff;margin-top:4px}.card small{color:#8da3bd}.section{font-size:11px;letter-spacing:2px;color:#90a6bf;margin:12px 0 15px}.empty{border:1px dashed #35455f;border-radius:12px;text-align:center;padding:34px 16px;color:#9cb0c8}.note{color:#8fa4bd;font-size:12px}.stButton button{border-radius:10px;min-height:42px}div[data-testid="stMetric"]{background:#111c2d;border:1px solid #24354d;border-radius:12px;padding:14px}div[data-testid="stMetricLabel"]{color:#9ab0c9}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>''', unsafe_allow_html=True)
st.markdown('''<div class="hero"><span class="eyebrow">CAMILO GUEVARA / LABORATORIO DE PROBABILIDAD</span>
<h1>Casino <span style="color:#46e3ba">Estadístico.</span></h1>
<p>El azar se juega. La estadística se comprende. Simula, encuentra eventos raros y observa cómo los datos se acercan a la teoría.</p>
<span class="badge">● MONTE CARLO &nbsp; / &nbsp; SIN DINERO REAL</span></div>''', unsafe_allow_html=True)

controls, main = st.columns([1, 3.4], gap='large')
with controls:
    st.markdown('<div class="section">01 / CONFIGURA EL EXPERIMENTO</div>', unsafe_allow_html=True)
    kind = st.radio('Mesa estadística', ['Binomial', 'Poisson', 'Normal'], horizontal=False,
                    help='Binomial: éxitos en n ensayos. Poisson: eventos por intervalo. Normal: observaciones continuas.')
    n, p, lam, mu, sigma = 100, .3, 4., 0., 1.
    if kind == 'Binomial':
        st.caption('LA RULETA BINOMIAL')
        n = st.slider('Ensayos por tirada (n)', 1, 500, 100)
        p = st.slider('Probabilidad de éxito (p)', 0., 1., .3, .01)
        st.caption(f'Cada tirada cuenta los éxitos de {n} ensayos Bernoulli independientes.')
    elif kind == 'Poisson':
        st.caption('EL DADO DE POISSON')
        lam = st.slider('Eventos esperados por intervalo (λ)', .1, 100., 4., .1)
        st.caption('Cada tirada representa un intervalo de duración fija. No es un dado de seis caras.')
    else:
        st.caption('LA MÁQUINA DE Z-SCORES')
        mu = st.slider('Media poblacional (μ)', -20., 20., 0., .5)
        sigma = st.slider('Desviación estándar (σ)', .1, 10., 1., .1)
    confidence = st.selectbox('Cobertura central / nivel del IC', [90, 95, 99], index=1) / 100
    batch = st.select_slider('Resultados por lote', [1, 10, 100, 1000], value=100)
    seed = int(st.number_input('Semilla reproducible', min_value=0, max_value=2147483647, value=2026, step=1))
    auto = st.toggle('Piloto automático', help='Añade un lote cada segundo mientras esta sesión está abierta. Límite: 20.000 resultados.')
    pause_rare = st.checkbox('Pausar al detectar un evento raro', value=False,
                              help='Pausa al terminar un lote con algún resultado de probabilidad bilateral menor que α. No altera ni descarta resultados.')
    st.caption('Cambiar modelo, parámetros o semilla reinicia la muestra. Cambiar la cobertura no la modifica.')

model = Model(kind, n, p, lam, mu, sigma)
config = (model, seed)
if st.session_state.get('config') != config:
    st.session_state.update({'config': config, 'rng': np.random.default_rng(seed), 'values': [], 'halted': False, 'reason': ''})
if not auto and st.session_state.get('was_auto', False):
    st.session_state.halted = False
    st.session_state.reason = ''
st.session_state.was_auto = auto

CYAN, GOLD, RED = '#46E3BA', '#ECC37A', '#FB7185'
def style(fig, height=370):
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', color='#9eb1c8', size=12), height=height,
        margin=dict(l=12,r=15,t=20,b=20), legend=dict(orientation='h',y=1.15,x=0,font=dict(size=10)),
        hovermode='x unified', transition_duration=250)
    fig.update_xaxes(gridcolor='#1d2a3e',zerolinecolor='#30425d')
    fig.update_yaxes(gridcolor='#1d2a3e',zerolinecolor='#30425d')
    return fig


def distribution_chart(a):
    x, y = model.grid(a)
    tail = model.tail_probability(x)
    alpha = 1-confidence
    fig = go.Figure()
    if kind == 'Normal':
        fig.add_trace(go.Scatter(x=x, y=y, name='Densidad teórica',line=dict(color=CYAN,width=3),fill='tozeroy',fillcolor='rgba(70,227,186,.08)'))
        for mask in (x < model.distribution.ppf(alpha/2), x > model.distribution.ppf(1-alpha/2)):
            fig.add_trace(go.Scatter(x=x[mask],y=y[mask],fill='tozeroy',line=dict(color=RED,width=1),fillcolor='rgba(251,113,133,.3)',showlegend=False,hoverinfo='skip'))
        if len(a): fig.add_trace(go.Histogram(x=a,histnorm='probability density',nbinsx=30,name='Muestra',marker_color=GOLD,opacity=.35))
    else:
        fig.add_trace(go.Bar(x=x,y=y,name='PMF teórica',marker_color=np.where(tail<alpha,RED,CYAN),opacity=.65))
        if len(a):
            v,c=np.unique(a,return_counts=True)
            fig.add_trace(go.Scatter(x=v,y=c/len(a),mode='lines+markers',name='Frecuencia observada',line=dict(color=GOLD,width=2)))
    mean, var = model.moments
    for k in (-2,-1,0,1,2):
        if var == 0 and k != 0: continue
        fig.add_vline(x=mean+k*np.sqrt(var),line_width=1,line_dash='dot',line_color='#43546f',annotation_text='μ' if k==0 else f'{k:+d}σ',annotation_position='top')
    if len(a):
        last=float(a[-1]); rare=float(model.tail_probability(last)) < alpha
        yy=model.distribution.pdf(last) if kind=='Normal' else model.distribution.pmf(last)
        fig.add_trace(go.Scatter(x=[last],y=[yy],mode='markers',name='Última tirada',marker=dict(color=RED if rare else '#FFFFFF',size=14,line=dict(color=RED if rare else CYAN,width=3))))
    fig.update_layout(xaxis_title='Resultado de una tirada',yaxis_title='Densidad' if kind=='Normal' else 'Probabilidad / frecuencia relativa',bargap=.2)
    return style(fig)


def add_batch():
    remaining = 20000-len(st.session_state['values'])
    if remaining <= 0:
        st.session_state.halted=True;st.session_state.reason='Límite de 20.000 resultados alcanzado.';return
    new = model.draw(st.session_state.rng, min(batch,remaining))
    st.session_state['values'].extend(new.tolist())
    if pause_rare and np.any(model.tail_probability(new)<1-confidence):
        st.session_state.halted=True;st.session_state.reason='Pausa educativa: el último lote contiene un evento raro.'

with main:
    @st.fragment(run_every=1.0 if auto else None)
    def dashboard():
        st.markdown('<div class="section">02 / OBSERVA EL AZAR EN ACCIÓN</div>', unsafe_allow_html=True)
        a1,a2,a3=st.columns([2,1,1])
        manual=a1.button(f'▶  Lanzar {batch:,} tiradas',type='primary',use_container_width=True,disabled=auto and not st.session_state.halted)
        reset=a2.button('↺ Reiniciar',use_container_width=True)
        resume=a3.button('Continuar',use_container_width=True,disabled=not st.session_state.halted)
        if reset:
            st.session_state.update(rng=np.random.default_rng(seed),values=[],halted=bool(auto),reason='Muestra reiniciada; pulsa Continuar para reactivar el piloto.' if auto else '')
        elif resume:
            st.session_state.halted=False;st.session_state.reason=''
        elif manual or (auto and not st.session_state.halted):
            add_batch()
        a=np.asarray(st.session_state['values'])
        mean,var=model.moments
        tabs=st.tabs(['Mesa en vivo','Inferencia','Cómo funciona','Datos y descargas'])
        with tabs[0]:
            plot,hud=st.columns([2.7,1],gap='medium')
            with plot:
                st.plotly_chart(distribution_chart(a),use_container_width=True,key='dist',config={'displayModeBar':False})
                st.caption('Las barras son probabilidades discretas; la curva normal es densidad. El rojo identifica colas bilaterales < α; no prueba sesgo por sí solo.')
            with hud:
                last_z=float(model.zscore(a[-1])) if len(a) else np.nan
                label='—' if np.isnan(last_z) else f'{last_z:+.2f} σ'
                st.markdown(f'<div class="card"><label>Z-SCORE / ÚLTIMA TIRADA</label><div class="number">{label}</div><small>Distancia respecto a μ en unidades de σ</small></div>',unsafe_allow_html=True)
                if var==0: st.caption('σ = 0: resultado determinista. Z no está definido.')
                st.metric('Tiradas acumuladas',f'{len(a):,}')
                st.metric('Media observada',f'{a.mean():.3f}' if len(a) else '—',f'Teórica: {mean:.3f}',delta_color='off')
                st.metric('Varianza muestral (s²)',f'{a.var(ddof=1):.3f}' if len(a)>1 else '—',f'Teórica: {var:.3f}',delta_color='off')
            if st.session_state.reason: st.info(st.session_state.reason)
            if len(a):
                tails=model.tail_probability(a); rare=tails<1-confidence; count=int(rare.sum())
                last_tail=float(tails[-1])
                (st.warning if rare[-1] else st.success)(f'Último resultado: {a[-1]:g} · Probabilidad bilateral de colas: {last_tail:.4%}. '+('Evento raro bajo el modelo seleccionado.' if rare[-1] else 'Dentro del comportamiento esperado bajo el modelo.'))
                st.caption(f'{count} eventos marcados de {len(a)} ({count/len(a):.2%}). α = {1-confidence:.0%}. Incluso un modelo correcto produce eventos raros; la tasa discreta puede ser menor que α.')
                lgn,log=st.columns([1.5,1])
                with lgn:
                    st.markdown('**La ley de los grandes números**')
                    ids=np.unique(np.linspace(0,len(a)-1,min(1000,len(a))).astype(int)); cumulative=np.cumsum(a)/np.arange(1,len(a)+1)
                    fig=go.Figure(go.Scatter(x=ids+1,y=cumulative[ids],line=dict(color=CYAN,width=2),name='Media acumulada'))
                    fig.add_hline(y=mean,line_color=GOLD,line_dash='dash',annotation_text='Media teórica')
                    fig.update_layout(xaxis_title='Número de tiradas',yaxis_title='Media acumulada')
                    st.plotly_chart(style(fig,250),use_container_width=True,key='lgn',config={'displayModeBar':False})
                    st.caption('Convergencia a largo plazo; el error no tiene que disminuir en cada tirada.')
                with log:
                    st.markdown('**Registro de eventos**')
                    f=export_frame(model,a).iloc[-8:][['tirada','resultado','z_score','probabilidad_colas']].iloc[::-1]
                    f['estado']=np.where(f.probabilidad_colas<1-confidence,'● Raro','○ Esperado')
                    st.dataframe(f,use_container_width=True,hide_index=True,column_config={'probabilidad_colas':st.column_config.NumberColumn('Colas',format='%.4f'),'z_score':st.column_config.NumberColumn('Z',format='%.2f')})
            else:
                st.markdown('<div class="empty">Tu experimento comienza aquí.<br><b>Lanza un lote o activa el piloto automático.</b><br>La distribución teórica ya está lista para compararse con tus datos.</div>',unsafe_allow_html=True)
        with tabs[1]:
            st.subheader('¿Un evento raro o evidencia sobre la media?')
            st.write('Un Z-score individual describe una tirada. El contraste siguiente utiliza la muestra completa para evaluar el parámetro del modelo.')
            result=mean_inference(model,a,confidence)
            if result:
                c1,c2,c3=st.columns(3)
                c1.metric(f'IC de la media · {confidence:.0%}',f'[{result["lower"]:.3f}, {result["upper"]:.3f}]')
                c2.metric('p-valor bilateral',f'{result["pvalue"]:.4g}')
                c3.metric('H₀: media del modelo',f'{mean:.3f}')
                st.info(('Se rechaza H₀' if result['pvalue'] < 1-confidence else 'No se rechaza H₀')+f' al nivel α = {1-confidence:.0%}, en este análisis de tamaño muestral fijo.')
                st.caption(result['method'])
                st.warning('Consultar el p-valor después de cada lote o detenerse cuando sea pequeño no controla el error global en α. Esta vista es exploratoria: fija el tamaño de muestra antes de hacer una prueba confirmatoria.')
                st.write('El p-valor no es la probabilidad de que H₀ sea verdadera. Un intervalo de confianza describe la cobertura del procedimiento al repetir el experimento, no la probabilidad de un parámetro fijo.')
            else: st.info('Genera datos para calcular el intervalo y el contraste.')
        with tabs[2]:
            st.markdown('''### Tres mesas, tres modelos
| Mesa | Una tirada representa | Media | Varianza |
|---|---|---|---|
| Binomial | Éxitos en n ensayos independientes con p constante | np | np(1−p) |
| Poisson | Eventos independientes en un intervalo a tasa constante | λ | λ |
| Normal | Una observación continua | μ | σ² |

**Dos niveles de repetición.** En Binomial, n es el número de ensayos dentro de una tirada. El tamaño del lote es cuántas tiradas independientes generas. 100 tiradas con n=200 representan 20.000 ensayos Bernoulli.

**Z = (x − μ) / σ.** En Normal, |Z|>1,96 corresponde aproximadamente al 5% bilateral. |Z|>2,5 corresponde al 1,24%, no a menos del 1%. Para 1% bilateral, el umbral es aproximadamente 2,576.

**Rareza discreta.** En Binomial y Poisson calculamos `min(1, 2 × min(P(X≤x), P(X≥x)))`. Incluimos el resultado observado en ambas colas; el criterio es conservador por la discreción. No convertimos Z a una probabilidad normal para estos modelos.

**Cobertura central ≠ intervalo de confianza de la media.** La primera describe resultados individuales; el segundo cuantifica incertidumbre de estimación. La pestaña Inferencia los distingue.

**Sin falsas alarmas narrativas.** Un evento raro no demuestra que el generador esté sesgado ni que la siguiente tirada vaya a compensarlo. Las tiradas son independientes.

**Reproducibilidad.** La misma semilla, parámetros y secuencia de lotes reconstruyen el experimento. Reiniciar restaura el generador; cambiar parámetros comienza otra muestra.

**Arquitectura.** Streamlit controla la sesión; `engine.py` implementa modelos, colas e inferencia con NumPy/SciPy; Plotly representa los resultados. Los datos se mantienen en la sesión, sin cuentas ni base de datos.
''')
            st.caption('Referencias: SciPy stats.binom, stats.poisson, stats.norm y stats.binomtest; documentación Streamlit st.fragment.')
        with tabs[3]:
            st.subheader('Un experimento que puedes auditar')
            st.json(dict(modelo=asdict(model),semilla=seed,tiradas=len(a),confianza=confidence))
            if len(a):
                f=export_frame(model,a)
                f['semilla']=seed;f['alpha']=1-confidence;f['evento_raro']=f.probabilidad_colas<1-confidence
                st.download_button('↓ Descargar resultados CSV',f.to_csv(index=False).encode('utf-8-sig'),'casino_estadistico.csv','text/csv')
                st.download_button('↓ Descargar configuración JSON',json.dumps(dict(modelo=asdict(model),semilla=seed,confianza=confidence,tiradas=len(a)),ensure_ascii=False,indent=2),'experimento.json','application/json')
                st.dataframe(f.tail(100),use_container_width=True,hide_index=True)
            else: st.info('El CSV estará disponible cuando generes la primera muestra.')
    dashboard()
st.divider()
st.caption('Casino Estadístico · Camilo Guevara · Proyecto académico de Sistemas y Ciencia de Datos · Python / NumPy / SciPy / Plotly / Streamlit')
