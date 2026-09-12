import React from 'react';
import {createRoot} from 'react-dom/client';
import LiveApp from './live/LiveApp';
import './styles.css';

class ErrorBoundary extends React.Component<{children:React.ReactNode},{error:string}> {
  state={error:''};
  static getDerivedStateFromError(error:Error){return {error:error.message};}
  render(){return this.state.error?<main className="login-note"><h1>界面异常</h1><p>{this.state.error}</p><p>F10 可暂停并松键；终端 Ctrl+C 停止服务。</p><button onClick={()=>location.reload()}>刷新</button></main>:this.props.children;}
}
createRoot(document.getElementById('root')!).render(<ErrorBoundary><LiveApp/></ErrorBoundary>);
