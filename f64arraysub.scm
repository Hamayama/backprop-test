;; -*- coding: utf-8 -*-
;;
;; f64arraysub.scm
;; 2019-3-11 v1.16
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
    f64array-ref          f64array-set!
    f64array-copy         f64array-copy!
    f64array-map          f64array-map!
    make-f64array-simple  make-f64array-same-shape
    f64array-simple
    f64array-nearly=?     f64array-nearly-zero?
    f64array-add-elements f64array-add-elements!
    f64array-sub-elements f64array-sub-elements!
    f64array-mul          f64array-mul!
    f64array-mul-elements f64array-mul-elements!
    f64array-sigmoid      f64array-sigmoid!
    f64array-relu         f64array-relu!
    f64array-step         f64array-step!
    f64array-transpose    f64array-transpose!
    f64array-row          f64array-row!
    f64array-col          f64array-col!
    ))
(select-module f64arraysub)

;; eigenmat モジュールのロード
;; (存在しなければ使用しない)
;(define *disable-eigenmat* #t) ; 無効化フラグ
(define *eigenmat-loaded*
  (and (not (global-variable-ref (current-module) '*disable-eigenmat* #f))
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

;; 行列の要素の参照(2次元のみ)
(define (f64array-ref A i j)
  (let ((is (s32vector-ref (slot-ref A 'start-vector) 0))
        (ie (s32vector-ref (slot-ref A 'end-vector)   0))
        (js (s32vector-ref (slot-ref A 'start-vector) 1))
        (je (s32vector-ref (slot-ref A 'end-vector)   1)))
    (unless (and (<= is i) (< i ie) (<= js j) (< j je))
      (error "invalid index value"))
    (f64vector-ref (slot-ref A 'backing-storage)
                   (+ (* (- i is) (- je js)) (- j js)))))

;; 行列の要素の設定(2次元のみ)
(define (f64array-set! A i j d)
  (let ((is (s32vector-ref (slot-ref A 'start-vector) 0))
        (ie (s32vector-ref (slot-ref A 'end-vector)   0))
        (js (s32vector-ref (slot-ref A 'start-vector) 1))
        (je (s32vector-ref (slot-ref A 'end-vector)   1)))
    (unless (and (<= is i) (< i ie) (<= js j) (< j je))
      (error "invalid index value"))
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

;; 行列のコピー(破壊的変更版)(タイプかサイズが違うときはエラー)
(define (array-copy! A B)
  (slot-set! A 'start-vector (slot-ref B 'start-vector))
  (slot-set! A 'end-vector   (slot-ref B 'end-vector))
  (slot-set! A 'mapper       (slot-ref B 'mapper))
  (let ((v1 (slot-ref A 'backing-storage))
        (v2 (slot-ref B 'backing-storage)))
    (cond
     ((and (vector? v1) (vector? v2)
           (= (vector-length v1) (vector-length v2)))
      (vector-copy! v1 0 v2))
     ((and (class-of v1) (class-of v2)
           (= (uvector-length v1) (uvector-length v2)))
      (uvector-copy! v1 0 v2))
     (else
      (error "can't copy array")))))

;; f64array を返す array-copy (エラーチェックなし)
(define (f64array-copy A)
  (if (eq? (class-of A) <f64array>)
    (array-copy A)
    (make <f64array>
      :start-vector    (slot-ref A 'start-vector)
      :end-vector      (slot-ref A 'end-vector)
      :mapper          (slot-ref A 'mapper)
      :backing-storage (coerce-to <f64vector> (slot-ref A 'backing-storage)))))

;; f64array を返す array-copy! (エラーチェックなし)
(define f64array-copy! array-copy!)

;; f64array を返す array-map (ただし shape の明示指定は不可)
(define (f64array-map proc ar0 . rest)
  (rlet1 ar (if (eq? (class-of ar0) <f64array>)
              (array-copy ar0)
              (make-f64array (array-shape ar0)))
    (apply array-map! ar proc ar0 rest)))

;; f64array を返す array-map! (エラーチェックなし)
(define f64array-map! array-map!)

;; 転置行列の生成(Gauche v0.9.7 の不具合対応(resの生成) + 高速化)
(define (%array-transpose a :optional (dim1 0) (dim2 1))
  (let* ([sh (array-copy (array-shape a))]
         [rank (array-rank a)]
         ;[tmp0 (array-ref sh dim1 0)]
         ;[tmp1 (array-ref sh dim1 1)])
         [vec  (slot-ref sh 'backing-storage)]
         [vs1  (* dim1 2)]
         [ve1  (+ vs1  1)]
         [vs2  (* dim2 2)]
         [ve2  (+ vs2  1)]
         [tmp0 (vector-ref vec vs1)]
         [tmp1 (vector-ref vec ve1)])
    ;(array-set! sh dim1 0 (array-ref sh dim2 0))
    ;(array-set! sh dim1 1 (array-ref sh dim2 1))
    ;(array-set! sh dim2 0 tmp0)
    ;(array-set! sh dim2 1 tmp1)
    (vector-set! vec vs1 (vector-ref vec vs2))
    (vector-set! vec ve1 (vector-ref vec ve2))
    (vector-set! vec vs2 tmp0)
    (vector-set! vec ve2 tmp1)
    ;(rlet1 res (array-copy a)
    (rlet1 res ((with-module gauche.array make-array-internal) (class-of a) sh)
      (array-for-each-index a
        (^[vec1] (let* ([vec2 (vector-copy vec1)]
                        [tmp (vector-ref vec2 dim1)])
                   (vector-set! vec2 dim1 (vector-ref vec2 dim2))
                   (vector-set! vec2 dim2 tmp)
                   (array-set! res vec2 (array-ref a vec1))))
        (make-vector rank)))))

;; 行列の次元数のチェック
(define-syntax check-array-rank
  (syntax-rules ()
    ((_ A)
     (unless (= (array-rank A) 2)
       (error "array rank must be 2")))
    ((_ A B ...)
     (unless (= (array-rank A) (array-rank B) ... 2)
       (error "array rank must be 2")))))

;; == 以下では、eigenmat モジュールがあれば使用する ==

;; 行列の生成(簡略版)(2次元のみ)
(define make-f64array-simple
  (if *eigenmat-loaded*
    (lambda (ns ne ms me . maybe-init)
      (rlet1 ar (eigen-make-array ns ne ms me)
        (unless (null? maybe-init)
          (f64vector-fill! (slot-ref ar 'backing-storage)
                           (car maybe-init)))))
    (lambda (ns ne ms me . maybe-init)
      (apply make-f64array (shape ns ne ms me) maybe-init))))

;; 同じ shape の行列の生成(簡略版)(2次元のみ)
(define make-f64array-same-shape
  (if *eigenmat-loaded*
    (lambda (A . maybe-init)
      (check-array-rank A)
      (let ((ns (array-start A 0))
            (ne (array-end   A 0))
            (ms (array-start A 1))
            (me (array-end   A 1)))
        (rlet1 ar (eigen-make-array ns ne ms me)
          (unless (null? maybe-init)
            (f64vector-fill! (slot-ref ar 'backing-storage)
                             (car maybe-init))))))
    (lambda (A . maybe-init)
      (check-array-rank A)
      (let ((ns (array-start A 0))
            (ne (array-end   A 0))
            (ms (array-start A 1))
            (me (array-end   A 1)))
        (apply make-f64array (shape ns ne ms me) maybe-init)))))

;; 行列の初期化データ付き生成(簡略版)(2次元のみ)
(define f64array-simple
  (if *eigenmat-loaded*
    eigen-array
    (lambda (ns ne ms me . inits)
      (rlet1 ar (make-f64array (shape ns ne ms me) 0)
        (f64vector-copy! (slot-ref ar 'backing-storage)
                         0 (list->f64vector inits))))))

;; 行列の一致チェック
(define f64array-nearly=?
  (if *eigenmat-loaded*
    eigen-array-nearly=?
    (lambda (ar1 ar2 :optional (precision 1e-12))
      (unless (= (array-rank ar1) (array-rank ar2))
        (error "array rank mismatch"))
      (dotimes (i (array-rank ar1))
        (unless (= (array-length ar1 i) (array-length ar2 i))
          (error "array shape mismatch")))
      (let ((v1    (slot-ref ar1 'backing-storage))
            (v2    (slot-ref ar2 'backing-storage))
            (norm1 0) (norm2 0) (norm3 0))
        (for-each
         (lambda (d1 d2)
           (inc! norm1 (* d1 d1))
           (inc! norm2 (* d2 d2))
           (inc! norm3 (* (- d1 d2) (- d1 d2))))
         v1 v2)
        (<= (%sqrt norm3) (* precision (min (%sqrt norm1) (%sqrt norm2))))))))

;; 行列のゼロチェック
(define f64array-nearly-zero?
  (if *eigenmat-loaded*
    eigen-array-nearly-zero?
    (lambda (ar1 :optional (precision 1e-12))
      (let ((v1    (slot-ref ar1 'backing-storage))
            (norm1 0))
        (for-each (lambda (d1) (inc! norm1 (* d1 d1))) v1)
        (<= (%sqrt norm1) precision)))))

;; 行列の和を計算
(define f64array-add-elements
  (if *eigenmat-loaded*
    (lambda (ar . rest) (fold-left eigen-array-add ar rest))
    (with-module gauche.array array-add-elements)))

;; 行列の和を計算(破壊的変更版)
;; (第1引数は結果を格納するためだけに使用)
(define f64array-add-elements!
  (if *eigenmat-loaded*
    (lambda (ar ar0 ar1 . rest)
      (eigen-array-add! ar ar0 ar1)
      (for-each (lambda (arX) (eigen-array-add! ar ar arX)) rest)
      ar)
    (lambda (ar . rest)
      (f64array-copy!
       ar
       (apply (with-module gauche.array array-add-elements) rest))
      ar)))

;; 行列の差を計算
(define f64array-sub-elements
  (if *eigenmat-loaded*
    (lambda (ar . rest) (fold-left eigen-array-sub ar rest))
    (with-module gauche.array array-sub-elements)))

;; 行列の差を計算(破壊的変更版)
;; (第1引数は結果を格納するためだけに使用)
(define f64array-sub-elements!
  (if *eigenmat-loaded*
    (lambda (ar ar0 ar1 . rest)
      (eigen-array-sub! ar ar0 ar1)
      (for-each (lambda (arX) (eigen-array-sub! ar ar arX)) rest)
      ar)
    (lambda (ar . rest)
      (f64array-copy!
       ar
       (apply (with-module gauche.array array-sub-elements) rest))
      ar)))

;; 行列の積を計算(2次元のみ)
(define f64array-mul
  (if *eigenmat-loaded*
    eigen-array-mul
    (with-module gauche.array array-mul)))

;; 行列の積を計算(破壊的変更版)(2次元のみ)
;; (第1引数は結果を格納するためだけに使用)
(define f64array-mul!
  (if *eigenmat-loaded*
    eigen-array-mul!
    (lambda (ar ar0 ar1)
      (f64array-copy! ar ((with-module gauche.array array-mul) ar0 ar1))
      ar)))

;; 行列の要素の積を計算
(define f64array-mul-elements
  (if *eigenmat-loaded*
    (lambda (ar . rest) (fold-left eigen-array-mul-elements ar rest))
    (with-module gauche.array array-mul-elements)))

;; 行列の要素の積を計算(破壊的変更版)
;; (第1引数は結果を格納するためだけに使用)
(define f64array-mul-elements!
  (if *eigenmat-loaded*
    (lambda (ar ar0 ar1 . rest)
      (eigen-array-mul-elements! ar ar0 ar1)
      (for-each (lambda (arX) (eigen-array-mul-elements! ar ar arX)) rest)
      ar)
    (lambda (ar . rest)
      (f64array-copy!
       ar
       (apply (with-module gauche.array array-mul-elements) rest))
      ar)))

;; 行列の要素に対して、シグモイド関数を計算
(define f64array-sigmoid
  (if *eigenmat-loaded*
    eigen-array-sigmoid
    (lambda (ar)
      (f64array-map
       (lambda (x1) (/. 1 (+ 1 (%exp (- x1))))) ; シグモイド関数
       ar))))

;; 行列の要素に対して、シグモイド関数を計算(破壊的変更版)
;; (第1引数は結果を格納するためだけに使用)
(define f64array-sigmoid!
  (if *eigenmat-loaded*
    eigen-array-sigmoid!
    (lambda (ar2 ar1)
      (f64array-map!
       ar2
       (lambda (x1) (/. 1 (+ 1 (%exp (- x1))))) ; シグモイド関数
       ar1)
      ar2)))

;; 行列の要素に対して、ReLU関数を計算
(define f64array-relu
  (if *eigenmat-loaded*
    eigen-array-relu
    (lambda (ar)
      (f64array-map
       (lambda (x1) (max 0 x1)) ; ReLU関数
       ar))))

;; 行列の要素に対して、ReLU関数を計算(破壊的変更版)
;; (第1引数は結果を格納するためだけに使用)
(define f64array-relu!
  (if *eigenmat-loaded*
    eigen-array-relu!
    (lambda (ar2 ar1)
      (f64array-map!
       ar2
       (lambda (x1) (max 0 x1)) ; ReLU関数
       ar1)
      ar2)))

;; 行列の要素に対して、ステップ関数を計算
(define f64array-step
  (if *eigenmat-loaded*
    eigen-array-step
    (lambda (ar)
      (f64array-map
       (lambda (x1) (if (> x1 0) 1 0)) ; ステップ関数
       ar))))

;; 行列の要素に対して、ステップ関数を計算(破壊的変更版)
;; (第1引数は結果を格納するためだけに使用)
(define f64array-step!
  (if *eigenmat-loaded*
    eigen-array-step!
    (lambda (ar2 ar1)
      (f64array-map!
       ar2
       (lambda (x1) (if (> x1 0) 1 0)) ; ステップ関数
       ar1)
      ar2)))

;; 転置行列を計算
(define f64array-transpose
  (if *eigenmat-loaded*
    eigen-array-transpose
    %array-transpose))

;; 転置行列を計算(破壊的変更版)
;; (第1引数は結果を格納するためだけに使用)
(define f64array-transpose!
  (if *eigenmat-loaded*
    eigen-array-transpose!
    (lambda (ar2 ar1)
      (f64array-copy! ar2 (%array-transpose ar1))
      ar2)))

;; 行列から行を抜き出す(2次元のみ)
(define f64array-row
  (if *eigenmat-loaded*
    eigen-array-row
    (lambda (ar1 i1)
      (check-array-rank ar1)
      (let ((n1 (array-length ar1 0))
            (m1 (array-length ar1 1))
            (is (array-start  ar1 0))
            (ie (array-end    ar1 0))
            (js (array-start  ar1 1)))
        (unless (and (>= i1 is) (< i1 ie))
          (error "invalid index value"))
        (let* ((ar2  (make-f64array (shape 0 1 0 m1))) ; 結果は 1 x m1 になる
               (vec2 (slot-ref ar2 'backing-storage)))
          (dotimes (j2 m1)
            (f64vector-set! vec2 j2 (f64array-ref ar1 i1 (+ j2 js))))
          ar2)))))

;; 行列から行を抜き出す(破壊的変更版)(2次元のみ)
;; (第1引数は結果を格納するためだけに使用)
(define f64array-row!
  (if *eigenmat-loaded*
    eigen-array-row!
    (lambda (ar2 ar1 i1)
      (check-array-rank ar1 ar2)
      (let ((n1   (array-length ar1 0))
            (m1   (array-length ar1 1))
            (is   (array-start  ar1 0))
            (ie   (array-end    ar1 0))
            (js   (array-start  ar1 1))
            (n2   (array-length ar2 0))
            (m2   (array-length ar2 1))
            (vec2 (slot-ref ar2 'backing-storage)))
        (unless (and (>= i1 is) (< i1 ie))
          (error "invalid index value"))
        (unless (and (= n2 1) (= m2 m1))               ; 結果は 1 x m1 になる
          (error "array shape mismatch"))
        (dotimes (j2 m1)
          (f64vector-set! vec2 j2 (f64array-ref ar1 i1 (+ j2 js))))
        ar2))))

;; 行列から列を抜き出す(2次元のみ)
(define f64array-col
  (if *eigenmat-loaded*
    eigen-array-col
    (lambda (ar1 j1)
      (check-array-rank ar1)
      (let ((n1 (array-length ar1 0))
            (m1 (array-length ar1 1))
            (is (array-start  ar1 0))
            (js (array-start  ar1 1))
            (je (array-end    ar1 1)))
        (unless (and (>= j1 js) (< j1 je))
          (error "invalid index value"))
        (let* ((ar2  (make-f64array (shape 0 n1 0 1))) ; 結果は n1 x 1 になる
               (vec2 (slot-ref ar2 'backing-storage)))
          (dotimes (i2 n1)
            (f64vector-set! vec2 i2 (f64array-ref ar1 (+ i2 is) j1)))
          ar2)))))

;; 行列から列を抜き出す(破壊的変更版)(2次元のみ)
;; (第1引数は結果を格納するためだけに使用)
(define f64array-col!
  (if *eigenmat-loaded*
    eigen-array-col!
    (lambda (ar2 ar1 j1)
      (check-array-rank ar1 ar2)
      (let ((n1   (array-length ar1 0))
            (m1   (array-length ar1 1))
            (is   (array-start  ar1 0))
            (js   (array-start  ar1 1))
            (je   (array-end    ar1 1))
            (n2   (array-length ar2 0))
            (m2   (array-length ar2 1))
            (vec2 (slot-ref ar2 'backing-storage)))
        (unless (and (>= j1 js) (< j1 je))
          (error "invalid index value"))
        (unless (and (= n2 n1) (= m2 1))               ; 結果は n1 x 1 になる
          (error "array shape mismatch"))
        (dotimes (i2 n1)
          (f64vector-set! vec2 i2 (f64array-ref ar1 (+ i2 is) j1)))
        ar2))))

