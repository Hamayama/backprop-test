;; -*- coding: utf-8 -*-
;;
;; backprop_main.scm
;; 2019-3-20 v2.20
;;
;; ＜内容＞
;;   Gauche を使って、バックプロパゲーションによる学習を行うプログラムです。
;;   出典は以下になります。
;;   「はじめてのディープラーニング」 我妻幸長 SB Creative 2018
;;     (「5.9 バックプロパゲーションの実装 -回帰-」)
;;
;;   ただし、高速化のために計算用の行列を追加しており、
;;   そのためオリジナルのプログラムよりも若干複雑になっています。
;;   また、中間層の数の設定と、活性化関数 (シグモイド / ReLU) の選択を可能にしています。
;;
;;   詳細については、以下のページを参照ください。
;;   https://github.com/Hamayama/backprop-test
;;
;; ＜使い方＞
;;   gosh backprop_main.scm paramfile
;;     paramfile : パラメータ設定ファイル
;;
(add-load-path "." :relative)
(use gauche.sequence)  ; shuffle
(use gauche.generator) ; generator->list
(use gauche.uvector)
(use gauche.array)
(use math.const)
(use data.random)      ; random-data-seed,reals-normal$
(use srfi-27)          ; random-source-randomize!,default-random-source
(use farrmat)          ; 2次元の f64array を扱う

;; 乱数の初期化
(set! (random-data-seed) (sys-time)) ; data.random (for reals-normal$)
(random-source-randomize! default-random-source) ; srfi-27 (for shuffle)
;; 正規分布の乱数ジェネレータ
(define gen-normal (reals-normal$))


(define outfile        "backprop_result.txt")   ; 出力ファイル名

(define input-data-0   (lrange 0 2pi 0.1))      ; 入力(リスト)
(define correct-data-0 (map %sin input-data-0)) ; 正解(リスト)
(define n-data         (length input-data-0))   ; データ数

(define input-data     (apply f-array           ; 入力(行列(1 x n-data))
                              0 1 0 n-data
                              ;; (入力の範囲を -1.0～1.0 に変換)
                              (map (lambda (x1) (/ (- x1 pi) pi)) input-data-0)))
(define correct-data   (apply f-array           ; 正解(行列(1 x n-data))
                              0 1 0 n-data
                              correct-data-0))

(define n-in           1)     ; 入力層のニューロン数
(define n-mid          20)    ; 中間層のニューロン数
(define n-out          1)     ; 出力層のニューロン数

(define ml-num         1)     ; 中間層の数
(define ml-func        'relu) ; 中間層の活性化関数(sigmoid / relu)

(define wb-width       0.01)  ; 重みとバイアスの幅
(define eta            0.1)   ; 学習係数
(define epoch          2001)  ; エポック数
(define interval       200)   ; 経過の表示間隔


;; 中間層クラス
(define-class <middle-layer> ()
  ((w      :init-value #f) ; 重み          (行列(n-upper x n))
   (b      :init-value #f) ; バイアス      (行列(1 x n))
   (x      :init-value #f) ; 入力          (行列(1 x n-upper))
   (y      :init-value #f) ; 出力          (行列(1 x n))
   (grad-w :init-value #f) ; 重みの勾配    (行列(サイズはwと同じ))
   (grad-b :init-value #f) ; バイアスの勾配(行列(サイズはbと同じ))
   (grad-x :init-value #f) ; 入力の勾配    (行列(サイズはxと同じ))
   (u      :init-value #f) ; 計算用        (行列(サイズはyと同じ))
   (delta  :init-value #f) ; 計算用        (行列(サイズはyと同じ))
   (tx     :init-value #f) ; 計算用        (行列(サイズはxの転置))
   (tw     :init-value #f) ; 計算用        (行列(サイズはwの転置))
   ))
(define (middle-layer-init ml n-upper n)
  (slot-set! ml 'w (apply f-array
                          0 n-upper 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal (* n-upper n)))))
  (slot-set! ml 'b (apply f-array
                          0 1 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal n))))
  (slot-set! ml 'x      (make-f-array 0 1 0 n-upper))
  (slot-set! ml 'y      (make-f-array 0 1 0 n))
  (slot-set! ml 'grad-w (make-f-array-same-shape (slot-ref ml 'w)))
  (slot-set! ml 'grad-b (make-f-array-same-shape (slot-ref ml 'b)))
  (slot-set! ml 'grad-x (make-f-array-same-shape (slot-ref ml 'x)))
  (slot-set! ml 'u      (make-f-array-same-shape (slot-ref ml 'y)))
  (slot-set! ml 'delta  (make-f-array-same-shape (slot-ref ml 'y)))
  (slot-set! ml 'tx     (make-f-array-same-shape (f-array-transpose (slot-ref ml 'x))))
  (slot-set! ml 'tw     (make-f-array-same-shape (f-array-transpose (slot-ref ml 'w))))
  )
(define (middle-layer-forward ml x)
  (slot-set! ml 'x x)
  ;; u = x * w + b
  (f-array-ab+c! (slot-ref ml 'u) x (slot-ref ml 'w) (slot-ref ml 'b))
  (cond
   ((eq? ml-func 'sigmoid)
    ;; シグモイド関数 : y = 1 / (1 + exp(-u))
    (f-array-sigmoid!
     (slot-ref ml 'y)
     (slot-ref ml 'u)))
   (else
    ;; ReLU関数 : y = max(0, u)
    (f-array-relu!
     (slot-ref ml 'y)
     (slot-ref ml 'u))))
  )
(define (middle-layer-backward ml grad-y)
  (cond
   ((eq? ml-func 'sigmoid)
    ;; シグモイド関数の微分 : delta = grad-y * (1 - y) * y
    (f-array-mul-elements!
     (slot-ref ml 'delta)
     (f-array-sub-elements! (slot-ref ml 'delta) (slot-ref ml 'y) 1)
     grad-y
     -1
     ;; (破壊的変更の関係で前に移動)
     ;(f-array-sub-elements! (slot-ref ml 'delta) (slot-ref ml 'y) 1)
     (slot-ref ml 'y)))
   (else
    ;; ReLU関数の微分 (ステップ関数) : delta = grad-y * (y > 0 ? 1 : 0)
    (f-array-mul-elements!
     (slot-ref ml 'delta)
     grad-y
     (f-array-step! (slot-ref ml 'delta) (slot-ref ml 'y)))))
  ;; grad-w = tx * delta
  (f-array-mul! (slot-ref ml 'grad-w)
                 (f-array-transpose! (slot-ref ml 'tx) (slot-ref ml 'x))
                 (slot-ref ml 'delta))
  ;; grad-b = delta
  (slot-set! ml 'grad-b (slot-ref ml 'delta))
  ;; grad-x = delta * tw
  (f-array-mul! (slot-ref ml 'grad-x)
                 (slot-ref ml 'delta)
                 (f-array-transpose! (slot-ref ml 'tw) (slot-ref ml 'w)))
  )
(define (middle-layer-update ml eta)
  ;; w -= eta * grad-w
  (f-array-ra+b! (slot-ref ml 'w) (- eta) (slot-ref ml 'grad-w) (slot-ref ml 'w))
  ;; b -= eta * grad-b
  (f-array-ra+b! (slot-ref ml 'b) (- eta) (slot-ref ml 'grad-b) (slot-ref ml 'b))
  )


;; 出力層クラス
(define-class <output-layer> ()
  ((w      :init-value #f) ; 重み          (行列(n-upper x n))
   (b      :init-value #f) ; バイアス      (行列(1 x n))
   (x      :init-value #f) ; 入力          (行列(1 x n-upper))
   (y      :init-value #f) ; 出力          (行列(1 x n))
   (grad-w :init-value #f) ; 重みの勾配    (行列(サイズはwと同じ))
   (grad-b :init-value #f) ; バイアスの勾配(行列(サイズはbと同じ))
   (grad-x :init-value #f) ; 入力の勾配    (行列(サイズはxと同じ))
   (u      :init-value #f) ; 計算用        (行列(サイズはyと同じ))
   (delta  :init-value #f) ; 計算用        (行列(サイズはyと同じ))
   (tx     :init-value #f) ; 計算用        (行列(サイズはxの転置))
   (tw     :init-value #f) ; 計算用        (行列(サイズはwの転置))
   (t      :init-value #f) ; 計算用        (行列(サイズはyと同じ))
   ))
(define (output-layer-init ol n-upper n)
  (slot-set! ol 'w (apply f-array
                          0 n-upper 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal (* n-upper n)))))
  (slot-set! ol 'b (apply f-array
                          0 1 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal n))))
  (slot-set! ol 'x      (make-f-array 0 1 0 n-upper))
  (slot-set! ol 'y      (make-f-array 0 1 0 n))
  (slot-set! ol 'grad-w (make-f-array-same-shape (slot-ref ol 'w)))
  (slot-set! ol 'grad-b (make-f-array-same-shape (slot-ref ol 'b)))
  (slot-set! ol 'grad-x (make-f-array-same-shape (slot-ref ol 'x)))
  (slot-set! ol 'u      (make-f-array-same-shape (slot-ref ol 'y)))
  (slot-set! ol 'delta  (make-f-array-same-shape (slot-ref ol 'y)))
  (slot-set! ol 'tx     (make-f-array-same-shape (f-array-transpose (slot-ref ol 'x))))
  (slot-set! ol 'tw     (make-f-array-same-shape (f-array-transpose (slot-ref ol 'w))))
  (slot-set! ol 't      (make-f-array-same-shape (slot-ref ol 'y)))
  )
(define (output-layer-forward ol x)
  (slot-set! ol 'x x)
  ;; u = x * w + b
  (f-array-ab+c! (slot-ref ol 'u) x (slot-ref ol 'w) (slot-ref ol 'b))
  ;; 恒等関数 : y = u
  (slot-set! ol 'y (slot-ref ol 'u))
  )
(define (output-layer-backward ol t)
  (slot-set! ol 't t)
  ;; delta = y - t
  (f-array-sub-elements! (slot-ref ol 'delta) (slot-ref ol 'y) t)
  ;; grad-w = tx * delta
  (f-array-mul! (slot-ref ol 'grad-w)
                 (f-array-transpose! (slot-ref ol 'tx) (slot-ref ol 'x))
                 (slot-ref ol 'delta))
  ;; grad-b = delta
  (slot-set! ol 'grad-b (slot-ref ol 'delta))
  ;; grad-x = delta * tw
  (f-array-mul! (slot-ref ol 'grad-x)
                 (slot-ref ol 'delta)
                 (f-array-transpose! (slot-ref ol 'tw) (slot-ref ol 'w)))
  )
(define (output-layer-update ol eta)
  ;; w -= eta * grad-w
  (f-array-ra+b! (slot-ref ol 'w) (- eta) (slot-ref ol 'grad-w) (slot-ref ol 'w))
  ;; b -= eta * grad-b
  (f-array-ra+b! (slot-ref ol 'b) (- eta) (slot-ref ol 'grad-b) (slot-ref ol 'b))
  )


;; メイン処理
(define (main args)
  (define paramfile (list-ref args 1 #f))
  (define mls       #f) ; 中間層(ベクタ)
  (define mls-rev   #f) ; 中間層の逆順(ベクタ)
  (define ol        #f) ; 出力層

  ;; パラメータ設定ファイルのロード
  (if paramfile (load paramfile))

  ;; 各層の生成
  (set! mls     (vector-tabulate ml-num (lambda (ml) (make <middle-layer>))))
  (set! mls-rev (list->vector (reverse (vector->list mls))))
  (set! ol      (make <output-layer>))

  ;; 各層の初期化
  (for-each-with-index
   (lambda (i ml)
     (if (= i 0)
       (middle-layer-init ml n-in  n-mid)
       (middle-layer-init ml n-mid n-mid)))
   mls)
  (output-layer-init ol n-mid n-out)

  ;; 学習
  (dotimes (i epoch)
    (let ((index-random (shuffle (iota n-data)))
          (total-error  0)
          (result       '()))
      (dolist (idx index-random)
        (let ((x (f-array-ref input-data   0 idx))
              (t (f-array-ref correct-data 0 idx))
              (y #f))
          ;; 順伝播
          (f-array-set! (slot-ref (vector-ref mls 0) 'x) 0 0 x)
          (for-each-with-index
           (lambda (i ml)
             (if (= i 0)
               (middle-layer-forward ml (slot-ref (vector-ref mls 0) 'x))
               (middle-layer-forward ml (slot-ref (vector-ref mls (- i 1)) 'y))))
           mls)
          (output-layer-forward  ol (slot-ref (vector-ref mls (- ml-num 1)) 'y))
          ;; 逆伝播 (ouput -> middle の順なので注意)
          (f-array-set! (slot-ref ol 't) 0 0 t)
          (output-layer-backward ol (slot-ref ol 't))
          (for-each-with-index
           (lambda (i ml)
             (if (= i 0)
               (middle-layer-backward ml (slot-ref ol 'grad-x))
               (middle-layer-backward ml (slot-ref (vector-ref mls (- ml-num i)) 'grad-x))))
           mls-rev)
          ;; 重みとバイアスの更新
          (for-each (lambda (ml) (middle-layer-update ml eta)) mls)
          (output-layer-update ol eta)
          ;; 結果の収集
          (when (= (modulo i interval) 0)
            (set! y (f-array-ref (slot-ref ol 'y) 0 0))
            (inc! total-error (* 0.5 (- y t) (- y t)))
            (push! result (cons x y)))
          ))
      ;; 結果の表示
      (when (= (modulo i interval) 0)
        ;(print (sort result < car))
        (print "Epoch: " i " / " epoch)
        (print "Error: " (/. total-error n-data)))
      ;; 最終結果((x,y)の組のデータ)の出力
      (when (= i (- epoch 1))
        (with-output-to-file outfile
          (lambda ()
            (for-each
             (lambda (r) (print (car r) "\t" (cdr r)))
             (sort result < car)))))
      ))

  ;; 最終パラメータの表示
  ;(for-each-with-index
  ; (lambda (i ml)
  ;   (format #t "ml~D.w = ~A~%" i (slot-ref (vector-ref mls i) 'w))
  ;   (format #t "ml~D.b = ~A~%" i (slot-ref (vector-ref mls i) 'b)))
  ; mls)
  ;(print "ol.w = "(slot-ref ol 'w))
  ;(print "ol.b = "(slot-ref ol 'b))

  0)

