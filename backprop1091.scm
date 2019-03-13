;; -*- coding: utf-8 -*-
;;
;; backprop1091.scm
;; 2019-3-13 v1.18
;;
;; ＜内容＞
;;   Gauche を使って、バックプロパゲーションによる学習を行うプログラムです。
;;   出典は以下になります。
;;   「はじめてのディープラーニング」 我妻幸長 SB Creative 2018
;;     (「5.9 バックプロパゲーションの実装 -回帰-」)
;;
;;   詳細については、以下のページを参照ください。
;;   https://github.com/Hamayama/backprop-test
;;
;; ＜流用ベース＞
;;   backprop1071.scm
;;
;; ＜変更点＞
;;   学習する関数を (2 / pi) * asin(sin(x)) にした (三角波)
;;
(add-load-path "." :relative)
(use gauche.sequence)  ; shuffle
(use gauche.generator) ; generator->list
(use gauche.uvector)
(use gauche.array)
(use math.const)
(use data.random)      ; random-data-seed,reals-normal$
(use srfi-27)          ; random-source-randomize!,default-random-source
(use f64arraysub)      ; 補助モジュール

;; 乱数の初期化
(set! (random-data-seed) (sys-time)) ; data.random (for reals-normal$)
(random-source-randomize! default-random-source) ; srfi-27 (for shuffle)
;; 正規分布の乱数ジェネレータ
(define gen-normal (reals-normal$))


(define outfile        "backprop_result1091.txt") ; 出力ファイル名

(define input-data-0   (lrange 0 2pi 0.1))      ; 入力(リスト)
(define correct-data-0 (map                     ; 正解(リスト)
                        (lambda (x1) (* 2 1/pi (%asin (%sin x1))))
                        input-data-0))
(define n-data         (length input-data-0))   ; データ数

(define input-data     (apply f64array-simple   ; 入力(行列(1 x n-data))
                              0 1 0 n-data
                              ;; (入力の範囲を -1.0～1.0 に変換)
                              (map (lambda (x1) (/ (- x1 pi) pi)) input-data-0)))
(define correct-data   (apply f64array-simple   ; 正解(行列(1 x n-data))
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
   (w0     :init-value #f) ; 計算用        (行列(サイズはwと同じ))
   (b0     :init-value #f) ; 計算用        (行列(サイズはbと同じ))
   (tx     :init-value #f) ; 計算用        (行列(サイズはxの転置))
   (tw     :init-value #f) ; 計算用        (行列(サイズはwの転置))
   ))
(define (middle-layer-init ml n-upper n)
  (slot-set! ml 'w (apply f64array-simple
                          0 n-upper 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal (* n-upper n)))))
  (slot-set! ml 'b (apply f64array-simple
                          0 1 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal n))))
  (slot-set! ml 'x      (make-f64array-simple 0 1 0 n-upper))
  (slot-set! ml 'y      (make-f64array-simple 0 1 0 n))
  (slot-set! ml 'grad-w (make-f64array-same-shape (slot-ref ml 'w)))
  (slot-set! ml 'grad-b (make-f64array-same-shape (slot-ref ml 'b)))
  (slot-set! ml 'grad-x (make-f64array-same-shape (slot-ref ml 'x)))
  (slot-set! ml 'u      (make-f64array-same-shape (slot-ref ml 'y)))
  (slot-set! ml 'delta  (make-f64array-same-shape (slot-ref ml 'y)))
  (slot-set! ml 'w0     (make-f64array-same-shape (slot-ref ml 'w)))
  (slot-set! ml 'b0     (make-f64array-same-shape (slot-ref ml 'b)))
  (slot-set! ml 'tx     (make-f64array-same-shape (f64array-transpose (slot-ref ml 'x))))
  (slot-set! ml 'tw     (make-f64array-same-shape (f64array-transpose (slot-ref ml 'w))))
  )
(define (middle-layer-forward ml x)
  (slot-set! ml 'x x)
  (slot-set! ml 'u (f64array-add-elements!
                    (slot-ref ml 'u)
                    (f64array-mul! (slot-ref ml 'u) x (slot-ref ml 'w))
                    (slot-ref ml 'b)))
  (slot-set! ml 'y
             (if (eq? ml-func 'sigmoid)
               (f64array-sigmoid!         ; シグモイド関数 ( 1/(1+exp(-u)) )
                (slot-ref ml 'y)
                (slot-ref ml 'u))
               (f64array-relu!            ; ReLU関数 ( (max 0 u) )
                (slot-ref ml 'y)
                (slot-ref ml 'u))))
  )
(define (middle-layer-backward ml grad-y)
  (slot-set! ml 'delta
             (if (eq? ml-func 'sigmoid)
               (f64array-mul-elements!    ; シグモイド関数の微分 ( grad-y*(1-y)*y )
                (slot-ref ml 'delta)
                (f64array-sub-elements! (slot-ref ml 'delta) (slot-ref ml 'y) 1)
                grad-y
                -1
                ;; (破壊的変更の関係で前に移動)
                ;(f64array-sub-elements! (slot-ref ml 'delta) (slot-ref ml 'y) 1)
                (slot-ref ml 'y))
               (f64array-mul-elements!    ; ReLU関数の微分 ( ステップ関数 )
                (slot-ref ml 'delta)
                grad-y
                (f64array-step! (slot-ref ml 'delta) (slot-ref ml 'y)))))
  (slot-set! ml 'grad-w (f64array-mul!
                         (slot-ref ml 'grad-w)
                         (f64array-transpose! (slot-ref ml 'tx) (slot-ref ml 'x))
                         (slot-ref ml 'delta)))
  (slot-set! ml 'grad-b (slot-ref ml 'delta))
  (slot-set! ml 'grad-x (f64array-mul!
                         (slot-ref ml 'grad-x)
                         (slot-ref ml 'delta)
                         (f64array-transpose! (slot-ref ml 'tw) (slot-ref ml 'w))))
  )
(define (middle-layer-update ml eta)
  (slot-set! ml 'w0 (f64array-mul-elements! (slot-ref ml 'w0) (slot-ref ml 'grad-w) eta))
  (slot-set! ml 'w  (f64array-sub-elements! (slot-ref ml 'w)  (slot-ref ml 'w) (slot-ref ml 'w0)))
  (slot-set! ml 'b0 (f64array-mul-elements! (slot-ref ml 'b0) (slot-ref ml 'grad-b) eta))
  (slot-set! ml 'b  (f64array-sub-elements! (slot-ref ml 'b)  (slot-ref ml 'b) (slot-ref ml 'b0)))
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
   (w0     :init-value #f) ; 計算用        (行列(サイズはwと同じ))
   (b0     :init-value #f) ; 計算用        (行列(サイズはbと同じ))
   (tx     :init-value #f) ; 計算用        (行列(サイズはxの転置))
   (tw     :init-value #f) ; 計算用        (行列(サイズはwの転置))
   (t      :init-value #f) ; 計算用        (行列(サイズはyと同じ))
   ))
(define (output-layer-init ol n-upper n)
  (slot-set! ol 'w (apply f64array-simple
                          0 n-upper 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal (* n-upper n)))))
  (slot-set! ol 'b (apply f64array-simple
                          0 1 0 n
                          (map (lambda (x1) (* wb-width x1))
                               (generator->list gen-normal n))))
  (slot-set! ol 'x      (make-f64array-simple 0 1 0 n-upper))
  (slot-set! ol 'y      (make-f64array-simple 0 1 0 n))
  (slot-set! ol 'grad-w (make-f64array-same-shape (slot-ref ol 'w)))
  (slot-set! ol 'grad-b (make-f64array-same-shape (slot-ref ol 'b)))
  (slot-set! ol 'grad-x (make-f64array-same-shape (slot-ref ol 'x)))
  (slot-set! ol 'u      (make-f64array-same-shape (slot-ref ol 'y)))
  (slot-set! ol 'delta  (make-f64array-same-shape (slot-ref ol 'y)))
  (slot-set! ol 'w0     (make-f64array-same-shape (slot-ref ol 'w)))
  (slot-set! ol 'b0     (make-f64array-same-shape (slot-ref ol 'b)))
  (slot-set! ol 'tx     (make-f64array-same-shape (f64array-transpose (slot-ref ol 'x))))
  (slot-set! ol 'tw     (make-f64array-same-shape (f64array-transpose (slot-ref ol 'w))))
  (slot-set! ol 't      (make-f64array-same-shape (slot-ref ol 'y)))
  )
(define (output-layer-forward ol x)
  (slot-set! ol 'x x)
  (slot-set! ol 'u (f64array-add-elements!
                    (slot-ref ol 'u)
                    (f64array-mul! (slot-ref ol 'u) x (slot-ref ol 'w))
                    (slot-ref ol 'b)))
  (slot-set! ol 'y (slot-ref ol 'u)) ; 恒等関数
  )
(define (output-layer-backward ol t)
  (slot-set! ol 't t)
  (slot-set! ol 'delta  (f64array-sub-elements! (slot-ref ol 'delta) (slot-ref ol 'y) t))
  (slot-set! ol 'grad-w (f64array-mul!
                         (slot-ref ol 'grad-w)
                         (f64array-transpose! (slot-ref ol 'tx) (slot-ref ol 'x))
                         (slot-ref ol 'delta)))
  (slot-set! ol 'grad-b (slot-ref ol 'delta))
  (slot-set! ol 'grad-x (f64array-mul!
                         (slot-ref ol 'grad-x)
                         (slot-ref ol 'delta)
                         (f64array-transpose! (slot-ref ol 'tw) (slot-ref ol 'w))))
  )
(define (output-layer-update ol eta)
  (slot-set! ol 'w0 (f64array-mul-elements! (slot-ref ol 'w0) (slot-ref ol 'grad-w) eta))
  (slot-set! ol 'w  (f64array-sub-elements! (slot-ref ol 'w)  (slot-ref ol 'w) (slot-ref ol 'w0)))
  (slot-set! ol 'b0 (f64array-mul-elements! (slot-ref ol 'b0) (slot-ref ol 'grad-b) eta))
  (slot-set! ol 'b  (f64array-sub-elements! (slot-ref ol 'b)  (slot-ref ol 'b) (slot-ref ol 'b0)))
  )


;; メイン処理
(define (main args)
  (define mls     (vector-tabulate ml-num (lambda (ml) (make <middle-layer>))))
  (define mls-rev (list->vector (reverse (vector->list mls))))
  (define ol      (make <output-layer>))

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
        (let ((x (f64array-ref input-data   0 idx))
              (t (f64array-ref correct-data 0 idx))
              (y #f))
          ;; 順伝播
          (f64array-set! (slot-ref (vector-ref mls 0) 'x) 0 0 x)
          (for-each-with-index
           (lambda (i ml)
             (if (= i 0)
               (middle-layer-forward ml (slot-ref (vector-ref mls 0) 'x))
               (middle-layer-forward ml (slot-ref (vector-ref mls (- i 1)) 'y))))
           mls)
          (output-layer-forward  ol (slot-ref (vector-ref mls (- ml-num 1)) 'y))
          ;; 逆伝播 (ouput -> middle の順なので注意)
          (f64array-set! (slot-ref ol 't) 0 0 t)
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
            (set! y (f64array-ref (slot-ref ol 'y) 0 0))
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

