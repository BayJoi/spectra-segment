import { type ButtonHTMLAttributes, type ReactNode } from "react";

type ButtonVariant = "default" | "ghost" | "accent";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: "sm";
  children?: ReactNode;
}

const variants: Record<ButtonVariant, string> = {
  default: "bg-neutral-800/50 border border-neutral-800/60 text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800 hover:border-neutral-700/60 hover:shadow-lg hover:shadow-black/20",
  ghost: "text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/50",
  accent: "bg-orange-500/15 border border-orange-500/30 text-orange-400 hover:bg-orange-500/25 hover:shadow-lg hover:shadow-orange-500/10",
};

const sizes = {
  sm: "h-7 px-2.5 text-[11px] gap-1.5",
};

export function Button({
  variant = "default",
  size = "sm",
  disabled,
  children,
  className = "",
  ...props
}: ButtonProps) {
  return (
    <button
      disabled={disabled}
      className={`inline-flex items-center justify-center rounded-lg font-medium whitespace-nowrap select-none cursor-pointer transition-all duration-200 hover:scale-105 active:scale-95 disabled:opacity-35 disabled:cursor-not-allowed disabled:scale-100 disabled:hover:scale-100 focus-visible:outline-2 focus-visible:outline-orange-500 focus-visible:outline-offset-2 ${variants[variant]} ${sizes[size]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
