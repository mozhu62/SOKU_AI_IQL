import {useEffect,useState} from 'react';

export type Row = Record<string, any>;
let requestSequence=0;
export async function api(path:string, init:RequestInit={}):Promise<any>{
  const response=await fetch(`/api/${path}`,{...init,headers:{'Content-Type':'application/json',...init.headers}});
  if(!response.ok){const error=await response.json().catch(()=>({detail:response.statusText}));throw new Error(error.detail||'请求失败');}
  return response.json();
}

export async function command(name:string,value:Row={}):Promise<Row>{
  // 局域网 HTTP 不一定提供 randomUUID；请求编号不是密码，不依赖安全上下文。
  const id=globalThis.crypto?.randomUUID?.()??`request_${Date.now().toString(36)}_${++requestSequence}_${Math.random().toString(36).slice(2)}`;
  let result=await api('commands',{method:'POST',body:JSON.stringify({id,name,value})});
  // 请求超时只表示仍在执行；不会换编号重复提交保存或训练任务。
  for(let tries=0;result.status==='queued'&&tries<240;tries++){
    await new Promise(resolve=>setTimeout(resolve,500));
    result=await api(`commands/${id}`);
  }
  if(result.status==='failed')throw new Error(result.message);
  return result;
}

export function useStatus(){
  const [state,setState]=useState<Row>({}),[connected,setConnected]=useState(false);
  useEffect(()=>{
    let socket:WebSocket|undefined,timer:ReturnType<typeof setTimeout>,cancelled=false;
    const connect=()=>{
      socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/api/stream`,['soku-iql']);
      socket.onopen=()=>setConnected(true);
      socket.onmessage=event=>{try{setState(JSON.parse(event.data));}catch{setConnected(false);}};
      socket.onclose=()=>{setConnected(false);if(!cancelled)timer=setTimeout(connect,1500);};
      socket.onerror=()=>socket?.close();
    };
    connect();return()=>{cancelled=true;clearTimeout(timer);socket?.close();};
  },[]);
  return {state,connected};
}

export function useQuery(path:string,interval=0){
  const [data,setData]=useState<Row|null>(null),[error,setError]=useState('');
  useEffect(()=>{
    let cancelled=false,timer:ReturnType<typeof setTimeout>|undefined;
    setData(null);setError('');
    const reload=async()=>{try{const value=await api(path);if(!cancelled){setData(value);setError('');}}
      catch(e){if(!cancelled)setError(String(e));}
      finally{if(!cancelled&&interval)timer=setTimeout(reload,interval);}};
    void reload();return()=>{cancelled=true;if(timer)clearTimeout(timer);};
  },[path,interval]);
  return {data,error};
}

export function useHistory(kind:string){
  const [rows,setRows]=useState<Row[]>([]),[error,setError]=useState('');
  useEffect(()=>{
    let cancelled=false,timer:ReturnType<typeof setTimeout>;
    const reload=async()=>{try{const data=await api(`history?kind=${kind}&limit=300`);if(!cancelled){setRows(data.rows);setError('');}}catch(e){if(!cancelled)setError(String(e));}finally{if(!cancelled)timer=setTimeout(reload,1000);}};
    void reload();return()=>{cancelled=true;clearTimeout(timer);};
  },[kind]);return {rows,error};
}
