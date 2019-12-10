- O:原始
- P:改进
- T:思考
---
>  在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
- O:
```
    public boolean Find(int target, int [][] array) {
        for(int [] cells : array){
            for(int cell : cells){
                if (target == cell){
                    return true;
                }
            }
        }
        return false;
    }
```
- P1:
```
	public boolean Find(int target, int [][] array) {
        int rows = array.length;
        int cols = array[0].length;
        int i = rows - 1, j = 0;//左下角元素坐标
        while (i >= 0 && j < cols) {//使其不超出数组范围
            if (target < array[i][j]) {
                i--;//查找的元素较少，往上找
            } else if (target > array[i][j]) {
                j++;//查找元素较大，往右找
            } else {
                return true;//找到
            }
        }
        return false;
    }
```
---
### 
> 请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
- O：
```
    public String replaceSpace(StringBuffer str) {
        if (str == null) {
            return null;
        }
        for (int i = 0; i < str.length(); i++) {
            char chr = str.charAt(i);
            if (chr == ' ') {
                str.delete(i, i + 1);
                str.insert(i, "%20");
                i = i + 2;
            }
        }
        return str.toString();
    }
```
---
>输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
- P1(递归):
```
	ArrayList<Integer> arrayList = new ArrayList<>();
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if (listNode == null){
            return arrayList;
        }
        this.printListFromTailToHead(listNode.next);
        arrayList.add(listNode.val);
        return arrayList;
    }
```
---
>输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
- O:
```
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre.length == 0 || in.length == 0) {
            return null;
        }
        TreeNode treeNode = new TreeNode(pre[0]);
        for (int i = 0; i < in.length; i++) {
            if (pre[0] == in[i]) {
                // copyOfRange 左闭右开
                treeNode.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, i + 1), Arrays.copyOfRange(in, 0, i));
                treeNode.right = reConstructBinaryTree(Arrays.copyOfRange(pre, i + 1, pre.length), Arrays.copyOfRange(in, i + 1, in.length));
            }
        }
        return null;
    }
```
---
> 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
- O:
```
Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();
    
     public void push(int node) {
        stack1.push(node);
    }

    public int pop() {
        while (!stack2.isEmpty()){
            return stack2.pop();
        }
        while (!stack1.isEmpty()){
            stack2.push(stack1.pop());
        }
        return stack2.pop();
    }
```
---
> 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
- O（封啸自创解法，真牛皮）:
```
public int minNumberInRotateArray(int[] array) {
        if (array.length == 0) {
            return 0;
        }
        int len = array.length;
        int low = 0;
        int high = len - 1;
        while (low < high) {
            int mid = low + high >> 1;
            if (array[mid] <= array[len - 1]) {//如果左侧小于右侧，
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        return array[low];
    }
```
- T:
1. 不一定非要考虑题中的旋转。通过求出最小亦是求出答案。
通过观察可知。如果array[mid]<=array[high] 可知，如果成立，那么符合二分查找。数据最小的一定在左边。其他情况。说明不成立。那么范围一定得从高位中找。
2. 
正数：r = 20 << 2;结果：r = 80

负数：r = -20 << 2;结果：r = -80

正数：r = 20 >> 2;结果：r = 5

负数：r = -20 >> 2;结果：r = -5
> 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
n<=39
- O:
```
	public int Fibonacci(int n) {
        if(n == 0){
            return 0;
        }else if(n == 1 ||n == 2){
            return 1;
        }else{
            return Fibonacci(n-1)+Fibonacci(n-2);
        }
    }
```
- P:
```
 	public int Fibonacci(int n) {
        if(n <= 1){
            return n;
        }
        int count = 0;
        int tempOne = 0;
        int tempTwo = 1;
        for(int i = 2;i<= n;i++){
            count = tempOne + tempTwo;
            tempOne = tempTwo;
            tempTwo = count;
        }
        return count;
    }
```
- T:
1.  斐波那契数列
            1                                x = 1
f(x) =   1       x = 2
           f(x - 1)  + f(x - 2)        x >= 3
> 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
- O:
```
	public int JumpFloor(int target) {
        if(target<=0){
            return 0;
        }
        if(target == 1){
            return 1;
        } 
        if(target == 2){
            return 2;
        } 
        return JumpFloor(target-1)+JumpFloor(target-2);
    }
```
- P:
```
public int JumpFloor(int target) {
        if(target<=0){
            return 0;
        }
        if(target == 1){
            return 1;
        } 
        if(target == 2){
            return 2;
        } 
        int count = 0;
        int resOne = 1;
        int resTwo = 2;
        for(int i = 2; i < target;i++){
            count = resOne + resTwo;
            resOne = resTwo;
            resTwo = count;
        }
        return count;
    }
```
> 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
- O:
```
	public int JumpFloorII(int target) {
        if(target<=0){
            return 0;
        }
        if(target == 1){
            return 1;
        } 
        if(target == 2){
            return 2;
        } 
        int count = 2;
        for(int i = 2;i<target;i++){
            count = 2* count;
        }
        return count;
    }
```
> 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
- O:
```
 public int RectCover(int target) {
        if(target == 0){
            return 0;
        }
        if(target == 1){
            return 1;
        }
        if(target == 2){
            return 2;
        }
        return RectCover(target-1)+RectCover(target-2);
    }
```
> 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
- O:
```

```
- T:
```
&按位与的运算规则是将两边的数转换为二进制位，然后运算最终值，运算规则即(两个为真才为真)1&1=1 , 1&0=0 , 0&1=0 , 0&0=0
```
><font color="#dd0000">输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。</font> 
- O:

> 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。保证base和exponent不同时为0

- O:
```
public double Power(double base, int exponent) {
        return Math.pow(base, exponent);
    }
```
- P:
```
	public double Power(double base, int exponent) {
        int temp = exponent>0? exponent : -exponent;
        double result = 1;
        for(int i = 0 ;i < temp; i++){
            result *= base;
        }
        return exponent > 0 ?  result : 1 / result;
    }
```
> 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
- O:
```
	public void reOrderArray(int [] array) {
        for(int i = 0;i< array.length;i++){
            for( int j = 0;j<array.length-i-1;j++ ){
                if(array[j]%2==0&&array[j+1]%2==1){
                    int temp = array[j];
                    array[j] = array[j+1];
                    array[j+1] = temp;
                }
            }
        }
    }
```
> 输入一个链表，输出该链表中倒数第k个结点。
- O:
```
	public ListNode FindKthToTail(ListNode head, int k) {
        ListNode pre = head;
        ListNode post = head;
        int prePos = 0;
        int postPos = 0;
        while (post != null) {
            post = post.next;
            postPos++;
            if (postPos - prePos > k) {
                prePos++;
                pre = pre.next;
            }
        }
        return postPos < k ? null : pre;
    }
```
> 输入一个链表，反转链表后，输出新链表的表头。
- O:
```
```
> 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
- O:
```

        ListNode head=new ListNode(-1);
        head.next=null;
        ListNode root=head;
        while(list1!=null&&list2!=null){
            if(list1.val<list2.val){
                head.next=list1;
                head=list1;
                list1=list1.next;
            }else{
                head.next=list2;
                head=list2;
                list2=list2.next;
            }
        }
        //把未结束的链表连接到合并后的链表尾部
        if(list1!=null){
            head.next=list1;
        }
        if(list2!=null){
            head.next=list2;
        }
        return root.next;
```
> 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
- O:
```
	public boolean isSubTree(TreeNode root1, TreeNode root2) {
        if (root2 == null) { // root2 为空时，说明已经比较完毕
            return true;
        }
        if (root1 == null) { // root1 为空时，说明root1长度不够
            return false;
        }
        if (root1.val == root2.val) {
            return isSubTree(root1.left, root2.left) && isSubTree(root1.right, root2.right);
        } else {
            return false;
        }
    }

    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null) {
            return false;
        }
        return isSubTree(root1, root2) || HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
    }

```
> 操作给定的二叉树，将其变换为源二叉树的镜像。
```
二叉树的镜像定义：源二叉树 
    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
    	镜像二叉树
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5
```
- O:
```
	public void Mirror(TreeNode root) {
         if (root!=null){
            TreeNode temp = root.left;
            root.left = root.right;
            root.right = temp;
            Mirror(root.left);
            Mirror(root.right);
        }
    }
```
- P1(非递归):
```
	public void Mirror(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<TreeNode>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode treeNode = stack.pop();
            if (treeNode.left != null || treeNode.right != null) {
                TreeNode temp = treeNode.left;
                treeNode.left = treeNode.right;
                treeNode.right = temp;
            }
            if (treeNode.left != null) {
                stack.push(treeNode.left);
            }
            if (treeNode.right != null) {
                stack.push(treeNode.right);
            }
        }
    }
```
> 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

- O:
```

```
> 定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
- O:
```

```
> 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
- O: 
```
public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA.length == 0 || popA.length == 0) {
            return false;
        }
        Stack<Integer> stack = new Stack<Integer>();
        int popIndex = 0;
        for (int i = 0; i < pushA.length; i++) {
            stack.push(pushA[i]);
            while (!stack.empty() && stack.peek() == popA[popIndex]) {
                stack.pop();
                popIndex++;
            }
        }
        return stack.empty();
    }
```
> 从上往下打印出二叉树的每个节点，同层节点从左至右打印。
- O:
```
public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> arrayList = new ArrayList<Integer>();
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        if (root == null) {
            return arrayList;
        }
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode treeNode = queue.poll();
            arrayList.add(treeNode.val);

            if (treeNode.left != null) {
                queue.add(treeNode.left);
            }
            if (treeNode.right != null) {
                queue.add(treeNode.right);
            }
        }
        return arrayList;
    }
```
> 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
- O:
```
	public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence.length == 0) {
            return false;
        }
        return isBst(sequence, 0, sequence.length - 1);
    }

    public boolean isBst(int[] sequence, int start, int end) {
        //终止条件
        if (start >= end) {
            return true;
        }

        // 寻找i的合适位置(从右边开始找)
        int j = end;
        while (j > start && sequence[j - 1] > sequence[end]) {
            --j;
        }

        // 判断是否都符合基本条件
        for (int i = j - 1; i >= start; i--) {
            if (sequence[i] > sequence[end]) {
                return false;
            }
        }

        return isBst(sequence, start, j - 1) && isBst(sequence, j, end - 1);
    }
```
> 输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)

- O:
```

```

> 输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
- O:
```

```

### 贪心算法总结：
总是在对问题求解时，作出看起来是当前是最好的选择。与之相对的是动态规划。
只进不退，贪心。能退能进，线性规划。