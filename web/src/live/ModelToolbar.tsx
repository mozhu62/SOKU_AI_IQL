import {useEffect, useState} from 'react';
import {FileCheck2, FolderOpen, RefreshCw, Upload, X} from 'lucide-react';
import {command, useQuery, type Row} from '../api';
import {Button} from '../components/ui/button';
import {ConfirmButton} from '../components/common';
import './model-toolbar.css';

const filename = (path: unknown) => String(path || '').split(/[\\/]/).pop() || '未选择';
const pathKey = (path: unknown) => String(path || '').replace(/\\/g, '/').toLowerCase();

export function ModelToolbar({state, connected}: {state: Row; connected: boolean}) {
  const [refresh, setRefresh] = useState(0);
  const {data, error: catalogError} = useQuery(`files?refresh=${refresh}`, 3000);
  const [selected, setSelected] = useState('');
  const [localBusy, setLocalBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const picked = state.selected_model as Row | undefined;
  const models: Row[] = [...(data?.models || [])];
  if (picked && !models.some(model => model.id === picked.id)) models.unshift(picked);
  const currentPath = state.checkpoint || state.runtime_config?.checkpoint;
  const current = models.find(model => pathKey(model.path) === pathKey(currentPath)
    || pathKey(`outputs/${model.name}`) === pathKey(currentPath));
  const selectedId = selected || current?.id || '';
  const candidate = models.find(model => model.id === selectedId);
  const picking = state.workbench_operation === 'browse_model';
  const disabled = !connected || localBusy || state.workbench_busy || state.state === 'initializing';
  const loaded = state.loaded && state.state !== 'stopped' && state.state !== 'error';

  // 原生窗口在服务端完成；刷新网页或请求等待超时也能从快照找回选中的文件。
  useEffect(() => {
    if (picked?.id) {
      setSelected(picked.id);
      setRefresh(value => value + 1);
    }
  }, [picked?.id]);

  const browse = async () => {
    setLocalBusy(true); setError('');
    setMessage('已请求暂停并打开 Windows 文件窗口；选择 .pt 后再点击加载模型。');
    try {
      const result = await command('browse_model');
      if (result.model) {setSelected(result.model.id); setRefresh(value => value + 1);}
      setMessage(result.status === 'queued' ? '文件窗口仍在打开，可继续选择或取消。' : result.message);
    } catch (reason) {setError(String(reason));}
    finally {setLocalBusy(false);}
  };

  const load = async () => {
    if (!candidate) throw new Error('请先选择模型');
    setLocalBusy(true); setError(''); setMessage(`正在检查并加载 ${candidate.filename || candidate.name}…`);
    try {
      const result = await command('load', {
        model_id: candidate.id, confirm_discard: true, pause_before_load: true,
      });
      setMessage(result.status === 'queued' ? '模型仍在加载，请等待顶部状态更新。' : result.message);
    } catch (reason) {
      setError(String(reason)); setMessage(''); throw reason;
    } finally {setLocalBusy(false);}
  };

  return <section className="model-toolbar" aria-label="模型选择与加载">
    <div className="model-current">
      <FileCheck2 size={19} className={loaded ? 'positive' : 'muted'}/>
      <div className="model-current-text">
        <small>{loaded ? '当前已加载模型 · Joint144 / TCN（窗口见时序状态）' : '尚无可运行模型'}</small>
        <strong title={String(currentPath || '')}>{loaded ? filename(currentPath) : '请选择并加载 IQL/BC 策略'}</strong>
        <span title={String(currentPath || '')}>{currentPath || '可直接选择桌面或其他磁盘的 .pt 文件'}</span>
      </div>
    </div>
    <div className="model-selection">
      <label htmlFor="live-model-file">待加载模型</label>
      <select id="live-model-file" value={selectedId} disabled={disabled}
        title={candidate?.path || '选择已登记模型，或点击右侧选择本机模型'}
        onChange={event => {setSelected(event.target.value); setMessage('已选择文件，点击加载模型生效'); setError('');}}>
        <option value="">选择已有模型 / 从本机浏览…</option>
        {models.map(model => <option key={model.id} value={model.id}>
          {model.name} · {(model.bytes / 1048576).toFixed(1)} MB{model.id === current?.id && loaded ? ' · 当前' : ''}
        </option>)}
      </select>
    </div>
    <div className="model-buttons">
      <Button variant="secondary" disabled={disabled} title="刷新已登记模型列表" aria-label="刷新模型列表"
        onClick={() => setRefresh(value => value + 1)}><RefreshCw size={15}/></Button>
      <Button disabled={disabled} onClick={() => void browse()}><FolderOpen size={16}/>选择本机模型</Button>
      <ConfirmButton disabled={disabled || !candidate} title={`加载 ${candidate?.filename || candidate?.name || '模型'}`}
        description="先暂停并松键，检查新模型通过后再切换。旧会话未完成小局保存为片段，模型文件不修改。加载后保持暂停，点击继续才会接管游戏。"
        onConfirm={load}><Upload size={16}/>加载模型</ConfirmButton>
    </div>
    {picking && <div className="model-picker-notice" role="status">
      <span>请在这台 Windows 电脑的文件窗口中选择 .pt；若窗口在后面，可用 Alt+Tab 切换。原模型已暂停。</span>
      <Button variant="ghost" onClick={async () => {
        try {const result = await command('cancel_model_selection'); setMessage(result.message);}
        catch (reason) {setError(String(reason));}
      }}><X size={14}/>取消选择</Button>
    </div>}
    {!picking && (error || message || catalogError) && <div className={`model-feedback ${error || catalogError ? 'error' : ''}`} role={error ? 'alert' : 'status'}>
      {error || message || catalogError}
    </div>}
    {candidate && <div className="model-selected-path" title={candidate.path}>待加载：{candidate.path || candidate.name}</div>}
  </section>;
}
