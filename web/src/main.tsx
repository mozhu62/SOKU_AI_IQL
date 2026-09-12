import React from 'react';
import {createRoot} from 'react-dom/client';
import App from './App';
import './styles.css';

class ErrorBoundary extends React.Component<{children:React.ReactNode},{error:string}> {
  state={error:''};
  static getDerivedStateFromError(error:Error){return {error:error.message};}
  render(){return this.state.error?<main className="login-note"><h1>界面异常</h1><p>{this.state.error}</p><p>训练线程独立运行，可在终端 Ctrl+C 保存退出。</p><button onClick={()=>location.reload()}>刷新</button></main>:this.props.children;}
}
createRoot(document.getElementById('root')!).render(<ErrorBoundary><App/></ErrorBoundary>);
