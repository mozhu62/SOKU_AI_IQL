import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
export function cn(...inputs: ClassValue[]) { return twMerge(clsx(inputs)); }
export type Row = Record<string, any>;
export const number = (v: unknown, digits = 3) => typeof v === 'number' && Number.isFinite(v) ? v.toLocaleString('zh-CN', { maximumFractionDigits: digits }) : '未记录';
export const percent = (v: unknown) => typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(2)}%` : '未记录';
export const signed = (v: unknown, digits = 3) => typeof v === 'number' && Number.isFinite(v) ? `${v >= 0 ? '+' : ''}${number(v, digits)}` : '未记录';
export const mean = (values: number[]) => values.length ? values.reduce((a,b) => a+b,0) / values.length : null;
