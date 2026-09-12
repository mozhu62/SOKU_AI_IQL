import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../../lib/utils';
const buttonVariants = cva('button inline-flex items-center justify-center gap-2 rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 disabled:pointer-events-none disabled:opacity-50', {
  variants: { variant: { default: 'button-primary', secondary: 'button-secondary', destructive: 'button-danger', ghost: 'button-ghost' } }, defaultVariants: { variant: 'default' }
});
export function Button({className, variant, asChild = false, ...props}: React.ComponentProps<'button'> & VariantProps<typeof buttonVariants> & {asChild?: boolean}) {
  const Comp = asChild ? Slot : 'button';
  return <Comp data-slot="button" className={cn(buttonVariants({variant,className}))} {...props}/>;
}
