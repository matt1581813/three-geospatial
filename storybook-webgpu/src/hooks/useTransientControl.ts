import type { Args } from '@storybook/react-vite'
import { getDefaultStore } from 'jotai'
import { useContext, useLayoutEffect, useRef } from 'react'
import shallowEqual from 'shallowequal'

import { StoryContext } from '../helpers/StoryContext'

export function useTransientControl<TArgs extends Args, const T>(
  selector: (args: TArgs) => T,
  onChange:
    | ((value: T, prevValue?: T) => void)
    | ((value: T, prevValue?: T) => () => void)
): void {
  const { argsAtom } = useContext(StoryContext)
  const store = getDefaultStore()

  const selectorRef = useRef(selector)
  selectorRef.current = selector
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange

  const initializedRef = useRef(false)
  const prevValueRef = useRef<T | undefined>(undefined)
  const cleanupRef = useRef<(() => void) | undefined>(undefined)

  useLayoutEffect(() => {
    const applyValue = (value: T): void => {
      if (initializedRef.current && shallowEqual(value, prevValueRef.current)) {
        return
      }

      cleanupRef.current?.()
      const result = onChangeRef.current(value, prevValueRef.current)
      cleanupRef.current = typeof result === 'function' ? result : undefined
      prevValueRef.current = value
      initializedRef.current = true
    }

    applyValue(selectorRef.current(store.get(argsAtom) as TArgs))

    const unsubscribe = store.sub(argsAtom, () => {
      applyValue(selectorRef.current(store.get(argsAtom) as TArgs))
    })

    return () => {
      unsubscribe()
      cleanupRef.current?.()
      cleanupRef.current = undefined
      prevValueRef.current = undefined
      initializedRef.current = false
    }
  }, [argsAtom, store])
}
