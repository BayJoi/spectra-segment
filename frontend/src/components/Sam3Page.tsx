import { useAtom } from "jotai";
import { Canvas } from "@/components/Canvas";
import { Sam3PromptBar } from "@/components/Sam3PromptBar";
import { uploadHoveredAtom } from "@/store/ui";

export function Sam3Page() {
  const [uploadHovered] = useAtom(uploadHoveredAtom);

  return (
    <div className="animate-fade-in absolute inset-0">
      <Canvas interactive={false} />
      <Sam3PromptBar />
      {uploadHovered && (
        <div className="absolute inset-0 z-30 pointer-events-none ring-2 ring-inset ring-orange-500/40" />
      )}
    </div>
  );
}
