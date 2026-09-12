import type { ReactNode } from 'react';
import { useState } from 'react';
import { Button } from './ui/button';
import { Dialog } from './ui/dialog';
import { number, type Row } from '../lib/utils';
export function Card({title,note,children,className=''}:{title:string;note?:string;children:ReactNode;className?:string}) {return <section className={`card ${className}`}><div className="card-heading"><h2>{title}</h2>{note&&<span className="muted">{note}</span>}</div>{children}</section>;}
export function Stat({title,value,note,tone=''}:{title:string;value:ReactNode;note?:string;tone?:string}) {return <div className={`stat ${tone}`}><div className="muted">{title}</div><strong>{value}</strong>{note&&<small>{note}</small>}</div>;}
export function DataTable({rows,columns,onSelect,selected}:{rows:Row[];columns:{key:string;title:string;render?:(v:any,row:Row)=>ReactNode}[];onSelect?:(r:Row)=>void;selected?:number}) {
 return <div className="table-scroll"><table><thead><tr>{columns.map(c=><th key={c.key}>{c.title}</th>)}</tr></thead><tbody>{rows.map((row,index)=><tr key={row.id??row.index??index} className={`${onSelect?'clickable':''} ${row.index===selected?'selected':''}`} onClick={()=>onSelect?.(row)}>{columns.map(c=><td key={c.key}>{c.render?c.render(row[c.key],row):row[c.key]===null||row[c.key]===undefined?'未记录':typeof row[c.key]==='number'?number(row[c.key]):String(row[c.key])}</td>)}</tr>)}</tbody></table>{!rows.length&&<div className="empty">没有符合条件的记录</div>}</div>;
}
export function ConfirmButton({title,description,children,onConfirm,disabled=false}:{title:string;description:string;children:ReactNode;onConfirm:()=>Promise<unknown>;disabled?:boolean}) {
 const [open,setOpen]=useState(false),[busy,setBusy]=useState(false),[error,setError]=useState('');
 return <><Button variant="secondary" disabled={disabled} onClick={()=>{setError('');setOpen(true);}}>{children}</Button><Dialog open={open} onOpenChange={v=>!busy&&setOpen(v)} title={title} description={description}>{error&&<p className="error">{error}</p>}<div className="dialog-actions"><Button variant="ghost" disabled={busy} onClick={()=>setOpen(false)}>取消</Button><Button disabled={busy} onClick={async()=>{setBusy(true);try{await onConfirm();setOpen(false);}catch(e){setError(String(e));}finally{setBusy(false);}}}>{busy?'提交中…':'确认'}</Button></div></Dialog></>;
}
