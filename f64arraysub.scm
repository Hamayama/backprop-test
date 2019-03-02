;; -*- coding: utf-8 -*-
;;
;; f64arraysub.scm
;; 2019-3-2 v1.01
;;
;; ＜内容＞
;;   Gauche で、2次元の f64array を扱うための補助的なモジュールです。
;;   gauche.array および eigenmat モジュールより後に
;;   use されることを想定しています。
;;
(define-module f64arraysub
  (use gauche.sequence)
  (use gauche.uvector)
  (use gauche.array)
  (use math.const)
  ;(use eigenmat)
  (define-module eigenmat)
  (import eigenmat)
  (export
    f64array-ref f64array-set! f64array-map
    make-f64array-simple f64array-simple
    f64array-add-elements f64array-sub-elements
    f64array-mul f64array-mul-elements
    f64array-sigmoid f64array-relu f64array-step
    f64array-transpose
    f64array-row f64array-col
    ))
(select-module f64arraysub)

;; eigenmat モジュールのロード
;; (存在しなければ使用しない)
(define *use-eigenmat-module* #t) ; 使用有無
(define *eigenmat-loaded*
  (and *use-eigenmat-module*
       (load "eigenmat" :error-if-not-found #f)))

;; shape の内部処理の高速化
(select-module gauche.array)
(define (shape->start/end-vector shape)
  (let* ([rank (array-end shape 0)]
         [cnt  (iota rank)]
         [vec  (slot-ref shape 'backing-storage)])
    ;(values (map-to <s32vector> (^i (array-ref shape i 0)) cnt)
    ;        (map-to <s32vector> (^i (array-ref shape i 1)) cnt))))
    (values (map-to <s32vector> (^i (vector-ref vec (* i 2))) cnt)
            (map-to <s32vector> (^i (vector-ref vec (+ (* i 2) 1))) cnt))))
(select-module f64arraysub)

;; 行列の情報取得(エラーチェックなし)
(define-inline (array-rank   A)
  (s32vector-length (slot-ref A 'start-vector)))
(define-inline (array-start  A dim)
  (s32vector-ref    (slot-ref A 'start-vector) dim))
(define-inline (array-end    A dim)
  (s32vector-ref    (slot-ref A 'end-vector)   dim))
(define-inline (array-length A dim)
  (- (s32vector-ref (slot-ref A 'end-vector)   dim)
     (s32vector-ref (slot-ref A 'start-vector) dim)))

;; 行列の要素の参照(エラーチェックなし)
(define (f64array-ref A i j)
  (let ((is (s32vector-ref (slot-ref A 'start-vector) 0))
        (js (s32vector-ref (slot-ref A 'start-vector) 1))
        (je (s32vector-ref (slot-ref A 'end-vector)   1)))
    (f64vector-ref (slot-ref A 'backing-storage)
                   (+ (* (- i is) (- je js)) (- j js)))))

;; 行列の要素の設定(エラーチェックなし)
(define (f64array-set! A i j d)
  (let ((is (s32vector-ref (slot-ref A 'start-vector) 0))
        (js (s32vector-ref (slot-ref A 'start-vector) 1))
        (je (s32vector-ref (slot-ref A 'end-vector)   1)))
    (f64vector-set! (slot-ref A 'backing-storage)
                    (+ (* (- i is) (- je js)) (- j js))
                    d)))

;; 行列のコピー(エラーチェックなし)
(define (array-copy A)
  (make (class-of A)
    :start-vector    (slot-ref A 'start-vector)
    :end-vector      (slot-ref A 'end-vector)
    :mapper          (slot-ref A 'mapper)
    :backing-storage (let1 v (slot-ref A 'backing-storage)
                       (if (vector? v)
                         (vector-copy v)
                         (uvector-copy v)))))

;; f64array を返す array-map (ただし shape の明示指定は不可)
(define (f64array-map proc ar0 . rest)
  (rlet1 ar (if (eq? (class-of ar0) <f64array>)
              (array-copy ar0)
              (make-f64array (array-shape ar0)))
    (apply array-map! ar proc ar0 rest)))

;; 転置行列の生成(Gauche v0.9.7 の不具合対応)
(define (%array-transpose a :optional (dim1 0) (dim2 1))
  (let* ([sh (array-copy (array-shape a))]
         [rank (array-rank a)]
         [tmp0 (array-ref sh dim1 0)]
         [tmp1 (array-ref sh dim1 1)])
    (array-set! sh dim1 0 (array-ref sh dim2 0))
    (array-set! sh dim1 1 (array-ref sh dim2 1))
    (array-set! sh dim2 0 tmp0)
    (array-set! sh dim2 1 tmp1)
    ;(rlet1 res (array-copy a)
    (rlet1 res ((with-module gauche.array make-array-internal) (class-of a) sh)
      (array-for-each-index a
        (^[vec1] (let* ([vec2 (vector-copy vec1)]
                        [tmp (vector-ref vec2 dim1)])
                   (vector-set! vec2 dim1 (vector-ref vec2 dim2))
                   (vector-set! vec2 dim2 tmp)
                   (array-set! res vec2 (array-ref a vec1))))
        (make-vector rank)))))

;; == 以下では、eigenmat モジュールがあれば使用する ==

;; 行列の生成
(define make-f64array-simple
  (if *eigenmat-loaded*
    (lambda (ns ne ms me . maybe-init)
      (rlet1 ar (eigen-make-array ns ne ms me)
        (unless (null? maybe-init)
          (f64vector-fill! (slot-ref ar 'backing-storage)
                           (car maybe-init)))))
    (lambda (ns ne ms me . maybe-init)
      (apply make-f64array (shape ns ne ms me) maybe-init))))

;; 行列の初期化データ付き生成
(define f64array-simple
  (if *eigenmat-loaded*
    eigen-array
    (lambda (ns ne ms me . inits)
      (apply f64array (shape ns ne ms me) inits))))

;; 行列の和を計算
(define f64array-add-elements
  (if *eigenmat-loaded*
    (lambda (ar0 . rest) (fold-left eigen-array-add ar0 rest))
    (with-module gauche.array array-add-elements)))

;; 行列の差を計算
(define f64array-sub-elements
  (if *eigenmat-loaded*
    (lambda (ar0 . rest) (fold-left eigen-array-sub ar0 rest))
    (with-module gauche.array array-sub-elements)))

;; 行列の積を計算
(define f64array-mul
  (if *eigenmat-loaded*
    eigen-array-mul
    (with-module gauche.array array-mul)))

;; 行列の要素の積を計算
(define f64array-mul-elements
  (if *eigenmat-loaded*
    (lambda (ar0 . rest) (fold-left eigen-array-mul-elements ar0 rest))
    (with-module gauche.array array-mul-elements)))

;; 行列の要素に対して、シグモイド関数を計算
(define f64array-sigmoid
  (if *eigenmat-loaded*
    eigen-array-sigmoid
    (lambda (ar)
      (f64array-map
       (lambda (x1) (/. 1 (+ 1 (%exp (- x1))))) ; シグモイド関数
       ar))))

;; 行列の要素に対して、ReLU関数を計算
(define f64array-relu
  (if *eigenmat-loaded*
    eigen-array-relu
    (lambda (ar)
      (f64array-map
       (lambda (x1) (max 0 x1)) ; ReLU関数
       ar))))

;; 行列の要素に対して、ステップ関数を計算
(define f64array-step
  (if *eigenmat-loaded*
    eigen-array-step
    (lambda (ar)
      (f64array-map
       (lambda (x1) (if (> x1 0) 1 0)) ; ステップ関数
       ar))))

;; 転置行列を計算
(define f64array-transpose
  (if *eigenmat-loaded*
    eigen-array-transpose
    %array-transpose))

;; 行列から行を抜き出す
(define f64array-row
  (if *eigenmat-loaded*
    eigen-array-row
    (lambda (ar1 i1)
      (let ((n1 (array-length ar1 0))
            (m1 (array-length ar1 1))
            (is (array-start  ar1 0))
            (ie (array-end    ar1 0))
            (js (array-start  ar1 1)))
        (unless (and (>= i1 is) (< i1 ie))
          (error "invalid index value"))
        (let* ((ar2  (make-f64array (shape 0 1 0 m1)))
               (vec2 (slot-ref ar2 'backing-storage)))
          (dotimes (j2 m1)
            (f64vector-set! vec2 j2 (f64array-ref ar1 i1 (+ j2 js))))
          ar2)))))

;; 行列から列を抜き出す
(define f64array-col
  (if *eigenmat-loaded*
    eigen-array-col
    (lambda (ar1 j1)
      (let ((n1 (array-length ar1 0))
            (m1 (array-length ar1 1))
            (is (array-start  ar1 0))
            (js (array-start  ar1 1))
            (je (array-end    ar1 1)))
        (unless (and (>= j1 js) (< j1 je))
          (error "invalid index value"))
        (let* ((ar2  (make-f64array (shape 0 n1 0 1)))
               (vec2 (slot-ref ar2 'backing-storage)))
          (dotimes (i2 n1)
            (f64vector-set! vec2 i2 (f64array-ref ar1 (+ i2 is) j1)))
          ar2)))))

