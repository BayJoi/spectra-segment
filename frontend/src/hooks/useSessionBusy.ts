import { useAtom } from "jotai";
import { imageEncodingAtom } from "@/store/ui";
import { matteBusyAtom } from "@/store/matte";
import { detectorLoadingAtom, isDetectingAtom, isSegmentingAtom } from "@/store/detection";

export function useSessionBusy() {
  const [imageEncoding] = useAtom(imageEncodingAtom);
  const [matteBusy] = useAtom(matteBusyAtom);
  const [isDetecting] = useAtom(isDetectingAtom);
  const [detectorLoading] = useAtom(detectorLoadingAtom);
  const [isSegmenting] = useAtom(isSegmentingAtom);

  const tasks: string[] = [];
  if (imageEncoding) tasks.push("an image upload");
  if (matteBusy) tasks.push("a matte run");
  if (isDetecting) tasks.push("a detection");
  if (detectorLoading) tasks.push("a detector load");
  if (isSegmenting) tasks.push("a batch segmentation");

  return { running: tasks.length > 0, tasks };
}
