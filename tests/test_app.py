from pathlib import Path
from streamlit.testing.v1 import AppTest
APP=str(Path(__file__).resolve().parents[1]/'app.py')
def test_modes_reset():
 at=AppTest.from_file(APP,default_timeout=30).run()
 assert not at.exception
 at.button[0].click().run()
 assert not at.exception and len(at.session_state['values'])==100
 original=list(at.session_state['values'])
 at.button[1].click().run()
 assert len(at.session_state['values'])==0
 at.button[0].click().run()
 assert at.session_state['values']==original
 for kind in ['Poisson','Normal']:
  at.radio[0].set_value(kind).run()
  assert not at.exception and len(at.session_state['values'])==0
  at.button[0].click().run()
  assert not at.exception and len(at.session_state['values'])==100
 at.selectbox[0].set_value(99).run()
 assert len(at.session_state['values'])==100

def test_auto_boundary():
 at=AppTest.from_file(APP,default_timeout=30).run()
 at.slider[1].set_value(0.).run();at.button[0].click().run()
 assert not at.exception and set(at.session_state['values'])=={0}
 at.toggle[0].set_value(True).run()
 assert not at.exception and len(at.session_state['values'])==200
 at.button[1].click().run()
 assert len(at.session_state['values'])==0 and at.session_state['halted']
